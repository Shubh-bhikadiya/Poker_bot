"""Discard Hold'em bot v2.

Same overall design as v1 (range-weighted equity, explicit three-card discard
model, per-leg opponent profile) with these upgrades:

* wall-clock time governor: every decision gets a deadline derived from the
  remaining time bank and remaining hands; all sampling loops are "anytime"
  and an emergency mode needs no sampling at all
* crash guard around every message handler
* memoised table-driven 7-card evaluator (about 4x faster than v1)
* turn/river: own discarded card is now excluded from the opponent's range
  and from the river deck (v1 bug); weights are cached and updated
  incrementally between requests
* flop: opponent range is self-normalised importance sampling over many
  uniformly drawn triples (v1 kept only the best of 4 candidates), opponent
  keep choice is averaged exactly instead of sampled, and the action equity
  is cross-validated between two sample halves to remove max-of-3 bias
* preflop: offline equity table (vs a random hand) used as a control variate
* richer raise sizing (incl. all-in) with an adverse-selection correction
  (the opponent calls with the better part of its range)

All learning stays in memory for one leg only. Standard library only.
"""

import itertools
import json
import math
import random
import sys
import time

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
SUIT_VALUE = {s: i for i, s in enumerate(SUITS)}
CARD_IDS = {
    r + s: ((RANK_VALUE[r] << 2) | SUIT_VALUE[s])
    for r in RANKS for s in SUITS
}
DECK = tuple(CARD_IDS.values())
STREETS = ("preflop", "flop", "turn", "river")
STREET_IDX = {s: i for i, s in enumerate(STREETS)}
RAISE_EQUITY_THRESHOLDS = {
    "preflop": 0.65,
    "flop": 0.66,
    "turn": 0.72,
    "river": 0.77,
}
perf = time.perf_counter


def card_rank(card):
    return card >> 2


def card_suit(card):
    return card & 3


def cards_from_strings(cards):
    return [CARD_IDS[c] for c in cards if c in CARD_IDS]


def _card_string(card):
    return RANKS[(card >> 2) - 2] + SUITS[card & 3]


# ----------------------------------------------------------------- evaluator

def straight_high(mask):
    for high in range(14, 5, -1):
        needed = 31 << (high - 4)
        if mask & needed == needed:
            return high
    wheel = (1 << 14) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2)
    return 5 if mask & wheel == wheel else 0


def _rank7_slow(cards):
    """Engine-compatible ranking tuple for 5 to 7 cards."""
    counts = [0] * 15
    suit_masks = [0] * 4
    rank_mask = 0
    for card in cards:
        rank = card >> 2
        suit = card & 3
        counts[rank] += 1
        suit_masks[suit] |= 1 << rank
        rank_mask |= 1 << rank

    flush_mask = 0
    for mask in suit_masks:
        if mask.bit_count() >= 5:
            flush_mask = mask
            sf = straight_high(mask)
            if sf:
                return (8, sf)

    quads = [r for r in range(14, 1, -1) if counts[r] == 4]
    if quads:
        q = quads[0]
        kicker = max(r for r in range(14, 1, -1) if r != q and counts[r])
        return (7, q, kicker)

    trips = [r for r in range(14, 1, -1) if counts[r] >= 3]
    if trips:
        t = trips[0]
        pairs = [r for r in range(14, 1, -1) if r != t and counts[r] >= 2]
        if pairs:
            return (6, t, pairs[0])

    if flush_mask:
        ranks = tuple(r for r in range(14, 1, -1) if flush_mask & (1 << r))[:5]
        return (5,) + ranks

    sh = straight_high(rank_mask)
    if sh:
        return (4, sh)

    if trips:
        t = trips[0]
        kickers = [r for r in range(14, 1, -1) if r != t and counts[r]][:2]
        return (3, t) + tuple(kickers)

    pairs = [r for r in range(14, 1, -1) if counts[r] >= 2]
    if len(pairs) >= 2:
        hi, lo = pairs[:2]
        kicker = max(r for r in range(14, 1, -1)
                     if r not in (hi, lo) and counts[r])
        return (2, hi, lo, kicker)
    if pairs:
        p = pairs[0]
        kickers = [r for r in range(14, 1, -1)
                   if r != p and counts[r]][:3]
        return (1, p) + tuple(kickers)

    return (0,) + tuple(r for r in range(14, 1, -1) if counts[r])[:5]


# Non-flush hands depend only on the rank multiset, which a base-5 digit sum
# identifies uniquely (a rank occurs at most four times). Flush detection uses
# base-8 suit counters.
_P5 = [0] * 64
_SK = [0] * 64
for _r in range(2, 15):
    for _s in range(4):
        _P5[(_r << 2) | _s] = 5 ** (_r - 2)
        _SK[(_r << 2) | _s] = 8 ** _s
_FS = [-1] * 4096
for _v in range(4096):
    for _suit in range(4):
        if (_v >> (3 * _suit)) & 7 >= 5:
            _FS[_v] = _suit
_NFT = {}   # rank multiset key -> tuple
_NFI = {}   # rank multiset key -> packed int


def _pack(t):
    v = 0
    for x in t:
        v = v * 15 + x
    for _ in range(6 - len(t)):
        v *= 15
    return v


def rank7(cards):
    """Ranking tuple (memoised for non-flush hands)."""
    k = 0
    s = 0
    for c in cards:
        k += _P5[c]
        s += _SK[c]
    if _FS[s] < 0:
        v = _NFT.get(k)
        if v is None:
            v = _NFT[k] = _rank7_slow(cards)
        return v
    return _rank7_slow(cards)


def ev7(cards):
    """Ranking as a single comparable integer (higher is better)."""
    k = 0
    s = 0
    for c in cards:
        k += _P5[c]
        s += _SK[c]
    if _FS[s] < 0:
        v = _NFI.get(k)
        if v is None:
            v = _NFI[k] = _pack(_rank7_slow(cards))
        return v
    return _pack(_rank7_slow(cards))


# ------------------------------------------------- preflop equity table plumbing

_PERMS = list(itertools.permutations(range(4)))


def canon_key(cards):
    """Suit-isomorphism class of a three-card hand as a short string."""
    rs = [(c >> 2, c & 3) for c in cards]
    best = None
    for p in _PERMS:
        t = tuple(sorted(((r, p[s]) for r, s in rs), reverse=True))
        if best is None or t < best:
            best = t
    return "".join(RANKS[r - 2] + str(s) for r, s in best)


PRE_EQ = {}


def _load_pre_table(data):
    for item in data.split():
        key, _, val = item.partition("=")
        if key and val:
            try:
                PRE_EQ[key] = int(val) / 1000.0
            except ValueError:
                pass


PRE_DATA = "222120=399 302120=479 302221=475 313020=474 313022=477 323130=435 402120=467 402221=465 403020=397 403021=399 403120=400 403121=386 403122=367 403130=514 403231=503 414020=520 414022=527 414030=514 414032=518 424140=471 502120=475 502221=472 503020=417 503021=402 503120=392 503121=392 503122=375 503130=533 503231=490 504020=410 504021=426 504030=424 504031=406 504120=413 504121=414 504122=392 504130=406 504131=412 504132=387 504140=549 504241=531 515020=559 515022=561 515030=548 515032=561 515040=536 515042=562 525150=505 602120=482 602221=475 603020=401 603021=388 603120=371 603121=414 603122=380 603130=508 603231=500 604020=409 604021=434 604030=416 604031=439 604120=410 604121=433 604122=390 604130=424 604131=430 604132=382 604140=528 604241=525 605020=439 605021=424 605030=439 605031=435 605040=443 605041=445 605120=434 605121=447 605122=417 605130=425 605131=451 605132=418 605140=443 605141=423 605142=411 605150=572 605251=573 616020=581 616022=557 616030=584 616032=587 616040=581 616042=600 616050=579 616052=606 626160=527 702120=477 702221=491 703020=417 703021=424 703120=412 703121=421 703122=399 703130=515 703231=517 704020=451 704021=435 704030=447 704031=423 704120=437 704121=436 704122=405 704130=429 704131=441 704132=425 704140=549 704241=552 705020=443 705021=440 705030=439 705031=449 705040=472 705041=472 705120=438 705121=451 705122=410 705130=465 705131=450 705132=415 705140=442 705141=448 705142=442 705150=573 705251=564 706020=461 706021=467 706030=453 706031=449 706040=446 706041=472 706050=443 706051=446 706120=441 706121=448 706122=437 706130=451 706131=457 706132=423 706140=452 706141=453 706142=432 706150=446 706151=466 706152=449 706160=615 706261=595 717020=612 717022=604 717030=592 717032=623 717040=616 717042=618 717050=634 717052=606 717060=628 717062=618 727170=582 802120=491 802221=493 803020=438 803021=419 803120=420 803121=412 803122=414 803130=502 803231=535 804020=447 804021=457 804030=449 804031=451 804120=441 804121=430 804122=419 804130=460 804131=449 804132=422 804140=556 804241=531 805020=455 805021=450 805030=445 805031=473 805040=470 805041=477 805120=461 805121=437 805122=439 805130=468 805131=465 805132=448 805140=464 805141=480 805142=442 805150=598 805251=599 806020=464 806021=455 806030=477 806031=480 806040=464 806041=478 806050=481 806051=474 806120=483 806121=469 806122=464 806130=447 806131=478 806132=457 806140=483 806141=477 806142=446 806150=460 806151=476 806152=467 806160=613 806261=585 807020=471 807021=502 807030=487 807031=487 807040=490 807041=469 807050=492 807051=488 807060=478 807061=485 807120=479 807121=467 807122=452 807130=492 807131=474 807132=471 807140=490 807141=492 807142=470 807150=487 807151=498 807152=482 807160=471 807161=491 807162=451 807170=608 807271=636 818020=660 818022=633 818030=640 818032=667 818040=669 818042=638 818050=649 818052=643 818060=631 818062=656 818070=666 818072=647 828180=635 902120=521 902221=497 903020=429 903021=436 903120=464 903121=438 903122=423 903130=526 903231=519 904020=419 904021=460 904030=453 904031=452 904120=442 904121=443 904122=420 904130=470 904131=453 904132=429 904140=554 904241=559 905020=464 905021=468 905030=462 905031=469 905040=480 905041=479 905120=447 905121=461 905122=452 905130=460 905131=462 905132=433 905140=482 905141=474 905142=480 905150=574 905251=576 906020=454 906021=478 906030=490 906031=480 906040=496 906041=484 906050=510 906051=513 906120=461 906121=481 906122=455 906130=489 906131=479 906132=460 906140=480 906141=483 906142=460 906150=484 906151=487 906152=466 906160=623 906261=609 907020=487 907021=483 907030=499 907031=500 907040=507 907041=489 907050=498 907051=497 907060=497 907061=503 907120=489 907121=480 907122=462 907130=477 907131=472 907132=479 907140=494 907141=490 907142=478 907150=517 907151=478 907152=481 907160=489 907161=489 907162=489 907170=652 907271=638 908020=502 908021=495 908030=548 908031=507 908040=495 908041=493 908050=518 908051=523 908060=517 908061=510 908070=515 908071=537 908120=496 908121=501 908122=498 908130=501 908131=487 908132=488 908140=520 908141=472 908142=475 908150=513 908151=507 908152=496 908160=519 908161=520 908162=475 908170=510 908171=506 908172=494 908180=668 908281=676 919020=677 919022=678 919030=690 919032=659 919040=684 919042=664 919050=684 919052=695 919060=680 919062=680 919070=688 919072=684 919080=672 919082=682 929190=659 A02120=531 A02221=502 A03020=533 A03021=545 A03120=534 A03121=541 A03122=537 A03130=571 A03231=541 A04020=547 A04021=547 A04030=552 A04031=551 A04120=550 A04121=542 A04122=557 A04130=549 A04131=561 A04132=538 A04140=588 A04241=576 A05020=557 A05021=544 A05030=547 A05031=561 A05040=574 A05041=563 A05120=566 A05121=533 A05122=535 A05130=550 A05131=557 A05132=541 A05140=546 A05141=549 A05142=548 A05150=611 A05251=603 A06020=575 A06021=559 A06030=572 A06031=571 A06040=573 A06041=575 A06050=573 A06051=593 A06120=567 A06121=562 A06122=546 A06130=569 A06131=562 A06132=561 A06140=571 A06141=569 A06142=540 A06150=575 A06151=548 A06152=570 A06160=627 A06261=625 A07020=560 A07021=591 A07030=586 A07031=597 A07040=599 A07041=582 A07050=594 A07051=572 A07060=581 A07061=594 A07120=592 A07121=558 A07122=545 A07130=574 A07131=563 A07132=557 A07140=578 A07141=577 A07142=583 A07150=600 A07151=592 A07152=562 A07160=580 A07161=583 A07162=539 A07170=660 A07271=654 A08020=604 A08021=595 A08030=590 A08031=595 A08040=587 A08041=570 A08050=607 A08051=617 A08060=576 A08061=587 A08070=614 A08071=606 A08120=608 A08121=603 A08122=565 A08130=582 A08131=584 A08132=602 A08140=602 A08141=567 A08142=587 A08150=602 A08151=584 A08152=579 A08160=590 A08161=583 A08162=572 A08170=618 A08171=606 A08172=574 A08180=693 A08281=692 A09020=598 A09021=581 A09030=586 A09031=603 A09040=597 A09041=600 A09050=614 A09051=600 A09060=597 A09061=622 A09070=609 A09071=611 A09080=611 A09081=588 A09120=609 A09121=608 A09122=567 A09130=614 A09131=582 A09132=592 A09140=624 A09141=596 A09142=602 A09150=616 A09151=594 A09152=608 A09160=611 A09161=592 A09162=589 A09170=636 A09171=618 A09172=584 A09180=596 A09181=586 A09182=579 A09190=704 A09291=716 A0J020=616 A0J021=617 A0J030=627 A0J031=633 A0J040=626 A0J041=630 A0J050=625 A0J051=632 A0J060=629 A0J061=627 A0J070=634 A0J071=625 A0J080=635 A0J081=647 A0J090=648 A0J091=643 A0J0T0=649 A0J0T1=646 A0J120=602 A0J121=609 A0J122=587 A0J130=622 A0J131=634 A0J132=615 A0J140=638 A0J141=623 A0J142=610 A0J150=630 A0J151=619 A0J152=625 A0J160=624 A0J161=633 A0J162=631 A0J170=637 A0J171=599 A0J172=622 A0J180=636 A0J181=623 A0J182=619 A0J190=648 A0J191=655 A0J192=629 A0J1J0=786 A0J1T0=646 A0J1T1=641 A0J1T2=636 A0J2J1=765 A0K020=634 A0K021=635 A0K030=634 A0K031=655 A0K040=642 A0K041=633 A0K050=656 A0K051=654 A0K060=660 A0K061=654 A0K070=649 A0K071=640 A0K080=684 A0K081=661 A0K090=666 A0K091=645 A0K0J0=689 A0K0J1=687 A0K0Q0=655 A0K0Q1=659 A0K0T0=665 A0K0T1=653 A0K120=631 A0K121=638 A0K122=615 A0K130=619 A0K131=641 A0K132=634 A0K140=648 A0K141=629 A0K142=611 A0K150=640 A0K151=656 A0K152=646 A0K160=645 A0K161=655 A0K162=638 A0K170=659 A0K171=666 A0K172=640 A0K180=664 A0K181=645 A0K182=633 A0K190=660 A0K191=666 A0K192=660 A0K1J0=666 A0K1J1=670 A0K1J2=658 A0K1K0=808 A0K1Q0=662 A0K1Q1=681 A0K1Q2=660 A0K1T0=673 A0K1T1=658 A0K1T2=650 A0K2K1=808 A0Q020=606 A0Q021=640 A0Q030=639 A0Q031=649 A0Q040=639 A0Q041=621 A0Q050=629 A0Q051=663 A0Q060=633 A0Q061=641 A0Q070=652 A0Q071=651 A0Q080=656 A0Q081=665 A0Q090=656 A0Q091=655 A0Q0J0=653 A0Q0J1=678 A0Q0T0=674 A0Q0T1=663 A0Q120=593 A0Q121=615 A0Q122=605 A0Q130=601 A0Q131=617 A0Q132=606 A0Q140=625 A0Q141=637 A0Q142=626 A0Q150=641 A0Q151=635 A0Q152=609 A0Q160=634 A0Q161=634 A0Q162=630 A0Q170=636 A0Q171=620 A0Q172=650 A0Q180=620 A0Q181=650 A0Q182=634 A0Q190=640 A0Q191=637 A0Q192=644 A0Q1J0=687 A0Q1J1=645 A0Q1J2=667 A0Q1Q0=786 A0Q1T0=658 A0Q1T1=663 A0Q1T2=663 A0Q2Q1=795 A0T020=612 A0T021=605 A0T030=608 A0T031=606 A0T040=629 A0T041=635 A0T050=626 A0T051=645 A0T060=619 A0T061=615 A0T070=618 A0T071=631 A0T080=630 A0T081=646 A0T090=630 A0T091=636 A0T120=634 A0T121=612 A0T122=576 A0T130=633 A0T131=612 A0T132=619 A0T140=630 A0T141=595 A0T142=631 A0T150=639 A0T151=603 A0T152=602 A0T160=613 A0T161=615 A0T162=631 A0T170=637 A0T171=617 A0T172=606 A0T180=641 A0T181=632 A0T182=610 A0T190=637 A0T191=624 A0T192=623 A0T1T0=750 A0T2T1=744 A1A020=832 A1A022=835 A1A030=823 A1A032=824 A1A040=827 A1A042=833 A1A050=827 A1A052=815 A1A060=834 A1A062=811 A1A070=810 A1A072=822 A1A080=834 A1A082=831 A1A090=810 A1A092=829 A1A0J0=834 A1A0J2=828 A1A0K0=824 A1A0K2=843 A1A0Q0=821 A1A0Q2=836 A1A0T0=816 A1A0T2=828 A2A1A0=820 J02120=500 J02221=492 J03020=472 J03021=461 J03120=458 J03121=454 J03122=469 J03130=540 J03231=512 J04020=476 J04021=498 J04030=470 J04031=493 J04120=455 J04121=470 J04122=470 J04130=496 J04131=503 J04132=471 J04140=568 J04241=558 J05020=481 J05021=494 J05030=489 J05031=487 J05040=496 J05041=497 J05120=477 J05121=481 J05122=461 J05130=485 J05131=474 J05132=471 J05140=487 J05141=516 J05142=496 J05150=582 J05251=587 J06020=490 J06021=501 J06030=502 J06031=511 J06040=507 J06041=532 J06050=528 J06051=514 J06120=486 J06121=478 J06122=461 J06130=485 J06131=531 J06132=471 J06140=497 J06141=490 J06142=491 J06150=518 J06151=501 J06152=504 J06160=639 J06261=615 J07020=496 J07021=491 J07030=506 J07031=520 J07040=519 J07041=522 J07050=521 J07051=514 J07060=521 J07061=541 J07120=502 J07121=522 J07122=481 J07130=507 J07131=516 J07132=502 J07140=538 J07141=511 J07142=512 J07150=529 J07151=532 J07152=506 J07160=527 J07161=547 J07162=517 J07170=632 J07271=641 J08020=532 J08021=553 J08030=518 J08031=538 J08040=543 J08041=538 J08050=537 J08051=546 J08060=537 J08061=572 J08070=548 J08071=554 J08120=533 J08121=519 J08122=509 J08130=538 J08131=516 J08132=488 J08140=547 J08141=530 J08142=513 J08150=530 J08151=525 J08152=545 J08160=531 J08161=550 J08162=519 J08170=538 J08171=570 J08172=526 J08180=683 J08281=665 J09020=536 J09021=548 J09030=552 J09031=546 J09040=530 J09041=547 J09050=559 J09051=547 J09060=550 J09061=563 J09070=566 J09071=575 J09080=546 J09081=569 J09120=542 J09121=523 J09122=524 J09130=538 J09131=541 J09132=507 J09140=531 J09141=549 J09142=528 J09150=552 J09151=556 J09152=534 J09160=561 J09161=554 J09162=532 J09170=542 J09171=575 J09172=545 J09180=551 J09181=541 J09182=532 J09190=704 J09291=704 J0T020=557 J0T021=553 J0T030=570 J0T031=565 J0T040=557 J0T041=563 J0T050=566 J0T051=588 J0T060=571 J0T061=576 J0T070=556 J0T071=598 J0T080=560 J0T081=575 J0T090=579 J0T091=574 J0T120=576 J0T121=554 J0T122=549 J0T130=557 J0T131=546 J0T132=539 J0T140=571 J0T141=549 J0T142=554 J0T150=577 J0T151=554 J0T152=534 J0T160=582 J0T161=577 J0T162=575 J0T170=565 J0T171=586 J0T172=566 J0T180=575 J0T181=578 J0T182=550 J0T190=592 J0T191=581 J0T192=555 J0T1T0=725 J0T2T1=726 J1J020=710 J1J022=738 J1J030=722 J1J032=723 J1J040=733 J1J042=759 J1J050=743 J1J052=734 J1J060=740 J1J062=741 J1J070=731 J1J072=741 J1J080=739 J1J082=735 J1J090=730 J1J092=743 J1J0T0=753 J1J0T2=736 J2J1J0=713 K02120=511 K02221=529 K03020=497 K03021=544 K03120=513 K03121=504 K03122=482 K03130=547 K03231=529 K04020=533 K04021=497 K04030=546 K04031=530 K04120=509 K04121=510 K04122=497 K04130=531 K04131=523 K04132=495 K04140=590 K04241=559 K05020=509 K05021=545 K05030=539 K05031=538 K05040=539 K05041=551 K05120=505 K05121=496 K05122=493 K05130=518 K05131=518 K05132=525 K05140=538 K05141=516 K05142=512 K05150=595 K05251=587 K06020=536 K06021=540 K06030=555 K06031=554 K06040=547 K06041=572 K06050=558 K06051=546 K06120=534 K06121=535 K06122=531 K06130=547 K06131=548 K06132=500 K06140=544 K06141=561 K06142=509 K06150=551 K06151=565 K06152=542 K06160=641 K06261=629 K07020=541 K07021=539 K07030=546 K07031=547 K07040=559 K07041=526 K07050=562 K07051=555 K07060=590 K07061=560 K07120=549 K07121=529 K07122=529 K07130=556 K07131=539 K07132=529 K07140=546 K07141=547 K07142=547 K07150=560 K07151=552 K07152=539 K07160=558 K07161=567 K07162=543 K07170=658 K07271=655 K08020=550 K08021=560 K08030=535 K08031=572 K08040=553 K08041=566 K08050=573 K08051=565 K08060=575 K08061=578 K08070=576 K08071=584 K08120=574 K08121=558 K08122=540 K08130=550 K08131=537 K08132=542 K08140=560 K08141=548 K08142=547 K08150=567 K08151=547 K08152=559 K08160=565 K08161=577 K08162=579 K08170=570 K08171=561 K08172=558 K08180=689 K08281=682 K09020=575 K09021=564 K09030=573 K09031=589 K09040=577 K09041=574 K09050=578 K09051=585 K09060=575 K09061=595 K09070=595 K09071=604 K09080=584 K09081=593 K09120=575 K09121=542 K09122=573 K09130=575 K09131=553 K09132=568 K09140=557 K09141=583 K09142=561 K09150=571 K09151=572 K09152=582 K09160=603 K09161=565 K09162=567 K09170=598 K09171=588 K09172=594 K09180=616 K09181=589 K09182=577 K09190=707 K09291=690 K0J020=610 K0J021=603 K0J030=587 K0J031=592 K0J040=614 K0J041=621 K0J050=612 K0J051=600 K0J060=620 K0J061=607 K0J070=628 K0J071=608 K0J080=615 K0J081=643 K0J090=624 K0J091=630 K0J0T0=617 K0J0T1=627 K0J120=585 K0J121=569 K0J122=589 K0J130=600 K0J131=591 K0J132=584 K0J140=619 K0J141=590 K0J142=593 K0J150=617 K0J151=592 K0J152=598 K0J160=599 K0J161=595 K0J162=584 K0J170=607 K0J171=606 K0J172=593 K0J180=614 K0J181=634 K0J182=599 K0J190=636 K0J191=613 K0J192=614 K0J1J0=777 K0J1T0=636 K0J1T1=617 K0J1T2=629 K0J2J1=755 K0Q020=609 K0Q021=608 K0Q030=593 K0Q031=599 K0Q040=614 K0Q041=629 K0Q050=599 K0Q051=623 K0Q060=622 K0Q061=620 K0Q070=637 K0Q071=629 K0Q080=619 K0Q081=631 K0Q090=608 K0Q091=628 K0Q0J0=635 K0Q0J1=627 K0Q0T0=640 K0Q0T1=631 K0Q120=603 K0Q121=605 K0Q122=582 K0Q130=610 K0Q131=618 K0Q132=591 K0Q140=599 K0Q141=593 K0Q142=599 K0Q150=613 K0Q151=614 K0Q152=616 K0Q160=644 K0Q161=614 K0Q162=629 K0Q170=608 K0Q171=620 K0Q172=594 K0Q180=629 K0Q181=622 K0Q182=619 K0Q190=636 K0Q191=645 K0Q192=605 K0Q1J0=624 K0Q1J1=621 K0Q1J2=619 K0Q1Q0=778 K0Q1T0=635 K0Q1T1=597 K0Q1T2=628 K0Q2Q1=784 K0T020=560 K0T021=594 K0T030=579 K0T031=592 K0T040=587 K0T041=609 K0T050=606 K0T051=618 K0T060=602 K0T061=615 K0T070=600 K0T071=612 K0T080=606 K0T081=619 K0T090=613 K0T091=612 K0T120=596 K0T121=553 K0T122=580 K0T130=583 K0T131=577 K0T132=563 K0T140=585 K0T141=591 K0T142=587 K0T150=604 K0T151=606 K0T152=605 K0T160=603 K0T161=584 K0T162=564 K0T170=596 K0T171=581 K0T172=592 K0T180=622 K0T181=615 K0T182=590 K0T190=609 K0T191=612 K0T192=605 K0T1T0=752 K0T2T1=743 K1K020=799 K1K022=787 K1K030=787 K1K032=788 K1K040=780 K1K042=809 K1K050=789 K1K052=796 K1K060=787 K1K062=794 K1K070=788 K1K072=805 K1K080=789 K1K082=801 K1K090=808 K1K092=790 K1K0J0=796 K1K0J2=808 K1K0Q0=774 K1K0Q2=804 K1K0T0=797 K1K0T2=814 K2K1K0=778 Q02120=503 Q02221=508 Q03020=473 Q03021=507 Q03120=493 Q03121=496 Q03122=466 Q03130=543 Q03231=531 Q04020=488 Q04021=504 Q04030=489 Q04031=527 Q04120=497 Q04121=499 Q04122=478 Q04130=500 Q04131=500 Q04132=492 Q04140=569 Q04241=562 Q05020=510 Q05021=514 Q05030=515 Q05031=511 Q05040=503 Q05041=503 Q05120=502 Q05121=505 Q05122=478 Q05130=506 Q05131=505 Q05132=515 Q05140=542 Q05141=506 Q05142=503 Q05150=588 Q05251=583 Q06020=516 Q06021=529 Q06030=525 Q06031=540 Q06040=516 Q06041=518 Q06050=516 Q06051=527 Q06120=515 Q06121=502 Q06122=495 Q06130=515 Q06131=509 Q06132=504 Q06140=536 Q06141=517 Q06142=525 Q06150=520 Q06151=558 Q06152=501 Q06160=624 Q06261=625 Q07020=524 Q07021=518 Q07030=527 Q07031=524 Q07040=523 Q07041=532 Q07050=516 Q07051=536 Q07060=556 Q07061=539 Q07120=539 Q07121=514 Q07122=471 Q07130=527 Q07131=521 Q07132=500 Q07140=549 Q07141=529 Q07142=521 Q07150=541 Q07151=554 Q07152=508 Q07160=554 Q07161=529 Q07162=525 Q07170=650 Q07271=670 Q08020=535 Q08021=547 Q08030=535 Q08031=541 Q08040=519 Q08041=562 Q08050=553 Q08051=539 Q08060=549 Q08061=555 Q08070=554 Q08071=562 Q08120=534 Q08121=543 Q08122=529 Q08130=535 Q08131=527 Q08132=528 Q08140=539 Q08141=546 Q08142=515 Q08150=552 Q08151=555 Q08152=529 Q08160=561 Q08161=562 Q08162=526 Q08170=553 Q08171=553 Q08172=523 Q08180=679 Q08281=684 Q09020=542 Q09021=539 Q09030=545 Q09031=573 Q09040=563 Q09041=559 Q09050=564 Q09051=581 Q09060=585 Q09061=572 Q09070=579 Q09071=563 Q09080=556 Q09081=590 Q09120=549 Q09121=543 Q09122=549 Q09130=551 Q09131=565 Q09132=536 Q09140=549 Q09141=551 Q09142=531 Q09150=579 Q09151=576 Q09152=559 Q09160=555 Q09161=578 Q09162=565 Q09170=564 Q09171=574 Q09172=569 Q09180=569 Q09181=586 Q09182=554 Q09190=703 Q09291=696 Q0J020=585 Q0J021=583 Q0J030=587 Q0J031=584 Q0J040=590 Q0J041=582 Q0J050=579 Q0J051=577 Q0J060=610 Q0J061=578 Q0J070=594 Q0J071=595 Q0J080=587 Q0J081=601 Q0J090=605 Q0J091=610 Q0J0T0=606 Q0J0T1=589 Q0J120=570 Q0J121=558 Q0J122=587 Q0J130=586 Q0J131=575 Q0J132=567 Q0J140=599 Q0J141=582 Q0J142=557 Q0J150=596 Q0J151=599 Q0J152=572 Q0J160=576 Q0J161=577 Q0J162=569 Q0J170=602 Q0J171=578 Q0J172=577 Q0J180=617 Q0J181=585 Q0J182=591 Q0J190=582 Q0J191=601 Q0J192=565 Q0J1J0=760 Q0J1T0=611 Q0J1T1=584 Q0J1T2=575 Q0J2J1=755 Q0T020=573 Q0T021=568 Q0T030=600 Q0T031=570 Q0T040=580 Q0T041=580 Q0T050=575 Q0T051=583 Q0T060=599 Q0T061=585 Q0T070=595 Q0T071=591 Q0T080=590 Q0T081=601 Q0T090=604 Q0T091=626 Q0T120=569 Q0T121=553 Q0T122=575 Q0T130=610 Q0T131=572 Q0T132=568 Q0T140=559 Q0T141=558 Q0T142=550 Q0T150=585 Q0T151=576 Q0T152=557 Q0T160=597 Q0T161=578 Q0T162=575 Q0T170=588 Q0T171=580 Q0T172=564 Q0T180=599 Q0T181=596 Q0T182=544 Q0T190=589 Q0T191=587 Q0T192=593 Q0T1T0=730 Q0T2T1=729 Q1Q020=752 Q1Q022=792 Q1Q030=770 Q1Q032=778 Q1Q040=765 Q1Q042=767 Q1Q050=751 Q1Q052=761 Q1Q060=772 Q1Q062=774 Q1Q070=765 Q1Q072=770 Q1Q080=759 Q1Q082=793 Q1Q090=766 Q1Q092=752 Q1Q0J0=745 Q1Q0J2=766 Q1Q0T0=768 Q1Q0T2=767 Q2Q1Q0=760 T02120=513 T02221=478 T03020=465 T03021=473 T03120=474 T03121=485 T03122=437 T03130=544 T03231=522 T04020=472 T04021=471 T04030=487 T04031=468 T04120=454 T04121=460 T04122=439 T04130=486 T04131=464 T04132=447 T04140=555 T04241=546 T05020=462 T05021=461 T05030=480 T05031=477 T05040=502 T05041=507 T05120=443 T05121=471 T05122=451 T05130=480 T05131=483 T05132=455 T05140=478 T05141=478 T05142=470 T05150=581 T05251=569 T06020=485 T06021=474 T06030=502 T06031=509 T06040=499 T06041=539 T06050=524 T06051=526 T06120=484 T06121=446 T06122=464 T06130=479 T06131=490 T06132=476 T06140=509 T06141=472 T06142=474 T06150=516 T06151=510 T06152=504 T06160=618 T06261=625 T07020=474 T07021=493 T07030=509 T07031=493 T07040=505 T07041=502 T07050=522 T07051=516 T07060=511 T07061=529 T07120=472 T07121=486 T07122=481 T07130=508 T07131=505 T07132=472 T07140=507 T07141=511 T07142=500 T07150=516 T07151=521 T07152=509 T07160=506 T07161=513 T07162=497 T07170=640 T07271=638 T08020=512 T08021=517 T08030=529 T08031=521 T08040=534 T08041=505 T08050=520 T08051=528 T08060=536 T08061=527 T08070=541 T08071=553 T08120=485 T08121=490 T08122=482 T08130=515 T08131=526 T08132=509 T08140=516 T08141=502 T08142=493 T08150=511 T08151=517 T08152=533 T08160=530 T08161=543 T08162=535 T08170=525 T08171=534 T08172=506 T08180=675 T08281=662 T09020=542 T09021=541 T09030=521 T09031=535 T09040=544 T09041=539 T09050=543 T09051=525 T09060=531 T09061=533 T09070=538 T09071=545 T09080=564 T09081=544 T09120=527 T09121=522 T09122=493 T09130=529 T09131=502 T09132=526 T09140=534 T09141=546 T09142=510 T09150=537 T09151=555 T09152=509 T09160=548 T09161=524 T09162=547 T09170=516 T09171=548 T09172=530 T09180=542 T09181=542 T09182=509 T09190=683 T09291=701 T1T020=712 T1T022=705 T1T030=711 T1T032=717 T1T040=718 T1T042=702 T1T050=711 T1T052=718 T1T060=710 T1T062=711 T1T070=714 T1T072=709 T1T080=721 T1T082=711 T1T090=708 T1T092=719 T2T1T0=687"
_load_pre_table(PRE_DATA)


def pre_table_equity(hole):
    if len(hole) != 3 or not PRE_EQ:
        return None
    return PRE_EQ.get(canon_key(hole))

def rank_strength(rank):
    """Coarse monotone strength feature for opponent-range likelihoods."""
    bases = (0.08, 0.29, 0.55, 0.73, 0.84, 0.88, 0.95, 0.985, 0.998)
    cat = rank[0]
    tail = rank[1:]
    frac = 0.0
    scale = 1.0
    for value in tail:
        frac += ((value - 2) / 12.0) / scale
        scale *= 15.0
    return min(0.999, bases[cat] + 0.075 * frac)

def preflop_strength(cards):
    ranks = [card_rank(c) for c in cards]
    best = 0.0
    for a, b in itertools.combinations(cards, 2):
        hi, lo = sorted((card_rank(a), card_rank(b)), reverse=True)
        gap = hi - lo
        score = 0.08 + 0.27 * ((hi - 2) / 12.0) + 0.15 * ((lo - 2) / 12.0)
        if hi == lo:
            score += 0.27 + 0.035 * ((hi - 2) / 12.0)
        else:
            if card_suit(a) == card_suit(b):
                score += 0.075
            if gap == 1:
                score += 0.065
            elif gap == 2:
                score += 0.04
            elif gap == 3:
                score += 0.018
            if hi >= 10 and lo >= 10:
                score += 0.03
        best = max(best, score)
    if len(set(ranks)) == 1:
        best += 0.13
    if len({card_suit(c) for c in cards}) == 1:
        best += 0.025
    if sorted(ranks)[2] - sorted(ranks)[0] <= 4:
        best += 0.025
    return min(0.96, max(0.04, best))


_SEQS = [frozenset(range(h - 4, h + 1)) for h in range(5, 15)]
_SEQS.append(frozenset((14, 2, 3, 4, 5)))


def pair_strength(pair, board):
    value = rank_strength(rank7(pair + board))
    if len(board) == 3:
        cards = pair + board
        sc = [0, 0, 0, 0]
        rankset = set()
        for c in cards:
            sc[c & 3] += 1
            rankset.add(c >> 2)
        if max(sc) == 4:
            value += 0.075
        outs = set()
        if value < 0.82:
            for seq in _SEQS:
                missing = seq - rankset
                if len(missing) == 1:
                    outs.update(missing)
        value += min(0.07, 0.022 * len(outs))
    return min(0.999, value)



def pair_quality(pair, board):
    return pair_strength(pair, board)


def best_pair(cards, board):
    pairs = list(itertools.combinations(cards, 2))
    return max(pairs, key=lambda p: pair_quality(list(p), board))


def simple_discard(cards, flop):
    """A quick, public-information-only discard policy for range simulation."""
    keep = best_pair(cards, flop)
    return next(c for c in cards if c not in keep)


def empty_model():
    return {
        "hands": 0,
        "pre": 0,
        "vpip": 0,
        "pfr": 0,
        "post_actions": 0,
        "post_raises": 0,
        "faced": 0,
        "folds": 0,
        "street_faced": {s: 0 for s in STREETS},
        "street_folds": {s: 0 for s in STREETS},
        "strength_observations": {},
        "discard_observations": 0,
        "simple_discard_matches": 0,
        "positions": {
            "button": {"pre": 0, "vpip": 0, "pfr": 0},
            "big_blind": {"pre": 0, "vpip": 0, "pfr": 0},
        },
        "recent": [],
        "notices": 0,
    }


def opponent_profile(model, opponent_position):
    vpip = (model["vpip"] + 2.0) / (model["pre"] + 4.0)
    pfr = (model["pfr"] + 1.0) / (model["pre"] + 4.0)
    aggression = (model["post_raises"] + 1.0) / (model["post_actions"] + 4.0)
    fold = (model["folds"] + 2.0) / (model["faced"] + 4.0)

    pos = model["positions"].get(opponent_position, {})
    if pos.get("pre", 0) >= 6:
        pos_vpip = (pos["vpip"] + 2.0) / (pos["pre"] + 4.0)
        pos_pfr = (pos["pfr"] + 1.0) / (pos["pre"] + 4.0)
        blend = min(0.35, pos["pre"] / 30.0)
        vpip = (1.0 - blend) * vpip + blend * pos_vpip
        pfr = (1.0 - blend) * pfr + blend * pos_pfr

    recent = model["recent"]
    if len(recent) >= 8:
        r_pre = sum(x["pre"] for x in recent)
        r_vpip = sum(x["vpip"] for x in recent)
        r_pfr = sum(x["pfr"] for x in recent)
        if r_pre:
            weight = min(0.40, len(recent) / 70.0)
            vpip = (1.0 - weight) * vpip + weight * (r_vpip + 2.0) / (r_pre + 4.0)
            pfr = (1.0 - weight) * pfr + weight * (r_pfr + 1.0) / (r_pre + 4.0)
        r_faced = sum(x["faced"] for x in recent)
        if r_faced >= 5:
            r_fold = sum(x["folds"] for x in recent)
            weight = min(0.35, r_faced / 50.0)
            fold = (1.0 - weight) * fold + weight * (r_fold + 2.0) / (r_faced + 4.0)

    street_fold = {}
    for street in STREETS:
        n = model["street_faced"].get(street, 0)
        f = model["street_folds"].get(street, 0)
        street_fold[street] = (f + 0.5 * 6.0) / (n + 6.0)
    return {
        "vpip": min(0.95, max(0.05, vpip)),
        "pfr": min(0.80, max(0.03, pfr)),
        "aggression": min(0.90, max(0.05, aggression)),
        "fold": min(0.90, max(0.10, fold)),
        "street_fold": street_fold,
        "strength_observations": model["strength_observations"],
        "discard_follow": (
            model["simple_discard_matches"] + 3.0
        ) / (model["discard_observations"] + 5.0),
    }

def sigmoid(x):
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))




# ------------------------------------------------------------ opponent model

def build_items(history):
    """Opponent's non-fold actions as (street_index, action, amount)."""
    out = []
    for it in history:
        if it.get("actor") != "opp":
            continue
        action = it.get("action", "")
        if action == "fold":
            continue
        si = STREET_IDX.get(it.get("street", "preflop"))
        if si is None:
            continue
        out.append((si, action, it.get("amount")))
    return out


def make_spec(item, profile):
    """Pre-resolve everything about an action that does not depend on the hand."""
    si, action, amount = item
    thr = None
    if si == 0:
        if action == "raise":
            thr = 0.58 - 0.45 * profile["pfr"]
        elif action == "call":
            thr = 0.66 - 0.52 * profile["vpip"]
    elif action == "raise":
        thr = 0.67 - 0.30 * profile["aggression"]
    obs = None
    o = profile.get("strength_observations", {}).get(STREETS[si] + "|" + action)
    if o and o[0] >= 6:
        count, s1, s2 = o
        mean = s1 / count
        var = max(0.0, s2 / count - mean * mean)
        obs = (mean, max(0.16, min(0.32, math.sqrt(var))))
    return (si, action, amount, thr, obs)


def spec_lik(strength, spec):
    """P(action | hand strength): same model as v1's action_likelihood."""
    si, action, amount, thr, obs = spec
    s = strength
    if s < 0.001:
        s = 0.001
    elif s > 0.999:
        s = 0.999
    if si == 0:
        if action == "raise":
            p = 0.04 + 0.96 * sigmoid((s - thr) / 0.13)
        elif action == "call":
            p = 0.06 + 0.94 * sigmoid((s - thr) / 0.14)
        else:
            p = 0.85 + 0.15 * (1.0 - s)
    else:
        if action == "raise":
            p = 0.08 + 0.92 * sigmoid((s - thr) / 0.18)
        elif action == "call":
            p = 0.20 + 0.80 * math.exp(-0.5 * ((s - 0.57) / 0.31) ** 2)
        elif action == "check":
            p = 0.32 + 0.68 * (1.0 - s) ** 0.85
        else:
            p = 0.5
        if action == "raise" and amount:
            if amount >= 35:
                p = p ** 1.10
            elif amount <= 6:
                p = 0.75 * p + 0.25 * 0.45
    if obs is not None:
        mean, sigma = obs
        emp = 0.08 + 0.92 * math.exp(-0.5 * ((s - mean) / sigma) ** 2)
        p = 0.78 * p + 0.22 * emp
    return 1.0 if p > 1.0 else (0.025 if p < 0.025 else p)


def hand_profile(model, state):
    p = state.get("profile")
    if p is None:
        p = opponent_profile(model, state.get("opp_position", "big_blind"))
        state["profile"] = p
    return p


def _pref_index(ps):
    if ps[0] >= ps[1] and ps[0] >= ps[2]:
        return 0
    return 1 if ps[1] >= ps[2] else 2


def _fpair(cache, x, y, flop):
    k = (x, y) if x < y else (y, x)
    v = cache.get(k)
    if v is None:
        v = cache[k] = pair_strength([x, y], flop)
    return v


# ------------------------------------------------------------ equity engines

def preflop_equity(hole, specs, profile, deadline, rng):
    """Equity before the flop.

    The offline table gives the equity against a random hand. Sampling then
    only has to estimate the shift caused by the opponent's actions, using
    the table as a control variate: eq = table + (weighted - uniform).
    """
    table = pre_table_equity(hole)
    pre_specs = [s for s in specs if s[0] == 0]
    if table is not None and not pre_specs:
        return table, 60.0
    blocked = set(hole)
    avail = [c for c in DECK if c not in blocked]
    follow = profile["discard_follow"]
    p_alt = (1.0 - follow) / 2.0
    sw = sw2 = swr = su = 0.0
    n = 0
    while n < 700:
        if n >= 20 and (n & 7) == 0 and perf() > deadline:
            break
        t = rng.sample(avail, 3)
        w = 1.0
        if pre_specs:
            pre = preflop_strength(t)
            for sp in pre_specs:
                w *= 0.22 + 0.78 * spec_lik(pre, sp)
            if w < 1e-5:
                w = 1e-5
        rest = [c for c in avail if c not in t]
        board = rng.sample(rest, 5)
        flop = board[:3]
        mine = ev7(list(best_pair(hole, flop)) + board)
        ks = ((t[0], t[1]), (t[0], t[2]), (t[1], t[2]))
        ps = [pair_strength(list(k), flop) for k in ks]
        pref = _pref_index(ps)
        r = 0.0
        for j in range(3):
            o = ev7(list(ks[j]) + board)
            res = 1.0 if mine > o else (0.5 if mine == o else 0.0)
            r += (follow if j == pref else p_alt) * res
        sw += w
        sw2 += w * w
        swr += w * r
        su += r
        n += 1
    if n == 0 or sw <= 0:
        return (table if table is not None else 0.5), 1.0
    ew = swr / sw
    eu = su / n
    eq = (table + (ew - eu)) if table is not None else ew
    return min(0.99, max(0.01, eq)), sw * sw / sw2


def flop_equities(hole, flop, specs, profile, deadline, rng,
                  min_n=30, max_n=900):
    """Equity of each of the three kept pairs, sharing every sampled world.

    Opponent triples are drawn uniformly and weighted by the range model
    (self-normalised importance sampling). The opponent's keep is averaged
    over its three options with the learned follow probability, and two
    random runouts are used per triple. Returns (pairs, values, halves,
    effective_samples); halves holds the estimates of the even and odd
    samples separately for cross-validated selection.
    """
    blocked = set(hole) | set(flop)
    avail = [c for c in DECK if c not in blocked]
    mine_pairs = [[hole[0], hole[1]], [hole[0], hole[2]], [hole[1], hole[2]]]
    follow = profile["discard_follow"]
    p_alt = (1.0 - follow) / 2.0
    need_pre = False
    for sp in specs:
        if sp[0] == 0:
            need_pre = True
            break
    S = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    W = [0.0, 0.0]
    W2 = [0.0, 0.0]
    n = 0
    while n < max_n:
        if n >= min_n and (n & 7) == 0 and perf() > deadline:
            break
        t = rng.sample(avail, 3)
        a, b, c = t
        ks = ((a, b), (a, c), (b, c))
        ps = [pair_strength([k[0], k[1]], flop) for k in ks]
        pref = _pref_index(ps)
        w = 1.0
        if specs:
            pre = preflop_strength(t) if need_pre else 0.0
            fs = ps[pref]
            for sp in specs:
                si = sp[0]
                if si == 0:
                    w *= 0.22 + 0.78 * spec_lik(pre, sp)
                elif si == 1:
                    w *= 0.22 + 0.78 * spec_lik(fs, sp)
            if w < 1e-5:
                w = 1e-5
        pk = [p_alt, p_alt, p_alt]
        pk[pref] = follow
        acc = [0.0, 0.0, 0.0]
        for _ in range(2):
            while True:
                x, y = rng.sample(avail, 2)
                if x not in t and y not in t:
                    break
            board = flop + [x, y]
            orank = [ev7([k[0], k[1]] + board) for k in ks]
            for i in range(3):
                m = ev7(mine_pairs[i] + board)
                e = 0.0
                for j in range(3):
                    o = orank[j]
                    e += pk[j] * (1.0 if m > o else (0.5 if m == o else 0.0))
                acc[i] += e
        h = n & 1
        S[h][0] += w * acc[0] * 0.5
        S[h][1] += w * acc[1] * 0.5
        S[h][2] += w * acc[2] * 0.5
        W[h] += w
        W2[h] += w * w
        n += 1
    wt = W[0] + W[1]
    if wt <= 0:
        return mine_pairs, [0.5, 0.5, 0.5], None, 1.0
    vals = [(S[0][i] + S[1][i]) / wt for i in range(3)]
    neff = wt * wt / (W2[0] + W2[1])
    halves = None
    if W[0] > 0 and W[1] > 0:
        halves = ([S[0][i] / W[0] for i in range(3)],
                  [S[1][i] / W[1] for i in range(3)])
    return mine_pairs, vals, halves, neff


def flop_action_equity(vals, halves):
    """Equity of the best pair without the upward max-of-three selection bias."""
    if halves is None:
        return max(vals)
    h0, h1 = halves
    b0 = max(range(3), key=h0.__getitem__)
    b1 = max(range(3), key=h1.__getitem__)
    return 0.5 * (h1[b0] + h0[b1])


def equity_uniform(hole, board, avail, deadline, rng):
    """Fallback when the opponent's discard is unknown (should not happen)."""
    river = len(board) >= 5
    wins = 0.0
    n = 0
    while n < 600 and len(avail) >= 3:
        if n >= 20 and perf() > deadline:
            break
        if river:
            a, b = rng.sample(avail, 2)
            bd = board
        else:
            a, b, r = rng.sample(avail, 3)
            bd = board + [r]
        m = ev7(hole + bd)
        o = ev7([a, b] + bd)
        wins += 1.0 if m > o else (0.5 if m == o else 0.0)
        n += 1
    return (wins / n if n else 0.5), float(max(n, 1))


def equity_post(hole, board, specs, profile, deadline, rng, state):
    """Turn and river equity against the range-weighted set of opponent pairs.

    Pairs are visited in a fixed random order until the deadline, so a
    truncated pass is an unbiased subsample. Weights are cached for the hand
    and only newly observed actions are applied on later requests.
    """
    kd = state.get("opp_discard")
    md = state.get("my_discard")
    blocked = set(hole) | set(board)
    if kd is not None:
        blocked.add(kd)
    if md is not None:
        blocked.add(md)
    avail = [c for c in DECK if c not in blocked]
    if kd is None or len(avail) < 3:
        return equity_uniform(hole, board, avail, deadline, rng)
    hc = state["hc"]
    flop = board[:3]
    availset = set(avail)
    order = hc.get("orderB")
    if order is None:
        order = list(itertools.combinations(avail, 2))
        rng.shuffle(order)
        hc["orderB"] = order
    pairs = [p for p in order if p[0] in availset and p[1] in availset]
    wB = hc.setdefault("wB", {})
    fcache = hc.setdefault("fs", {})
    follow = profile["discard_follow"]
    p_alt = (1.0 - follow) / 2.0
    nspec = len(specs)
    river = len(board) >= 5
    K = 6
    if river:
        mine = ev7(hole + board)
        rivers = my_by = None
        nr = 0
    else:
        rivers = avail[:]
        rng.shuffle(rivers)
        nr = len(rivers)
        my_by = {r: ev7(hole + board + [r]) for r in rivers}
    W = W2 = acc = 0.0
    n = 0
    b4 = board[:4]
    b5 = board[:5]
    for idx, pr in enumerate(pairs):
        if n >= 40 and (n & 15) == 0 and perf() > deadline:
            break
        a, b = pr
        e = wB.get(pr)
        if e is None:
            pre = preflop_strength((a, b, kd))
            f_ab = _fpair(fcache, a, b, flop)
            f_ad = _fpair(fcache, a, kd, flop)
            f_bd = _fpair(fcache, b, kd, flop)
            if f_ab >= f_ad and f_ab >= f_bd:
                predicted = kd
            elif f_ad >= f_bd:
                predicted = b
            else:
                predicted = a
            p_d = follow if predicted == kd else p_alt
            e = [0.25 + 0.75 * p_d, 0, pre, f_ab]
            wB[pr] = e
        if e[1] < nspec:
            w = e[0]
            for i in range(e[1], nspec):
                sp = specs[i]
                si = sp[0]
                if si == 0:
                    st = e[2]
                elif si == 1:
                    st = e[3]
                elif si == 2:
                    if len(board) < 4:
                        continue
                    st = pair_strength([a, b], b4)
                else:
                    if len(board) < 5:
                        continue
                    st = pair_strength([a, b], b5)
                w *= 0.22 + 0.78 * spec_lik(st, sp)
            e[0] = w
            e[1] = nspec
        w = e[0] if e[0] > 1e-5 else 1e-5
        if river:
            o = ev7([a, b] + board)
            res = 1.0 if mine > o else (0.5 if mine == o else 0.0)
        else:
            res = 0.0
            cnt = 0
            base = idx * K
            for t in range(K + 2):
                r = rivers[(base + t) % nr]
                if r == a or r == b:
                    continue
                o = ev7([a, b] + board + [r])
                m = my_by[r]
                res += 1.0 if m > o else (0.5 if m == o else 0.0)
                cnt += 1
                if cnt == K:
                    break
            if cnt == 0:
                continue
            res /= cnt
        W += w
        W2 += w * w
        acc += w * res
        n += 1
    if W <= 0:
        return 0.5, 1.0
    return acc / W, W * W / W2


def quick_equity(hole, board):
    """Sampling-free estimate for emergency mode."""
    if not board:
        t = pre_table_equity(hole)
        return t if t is not None else 0.5
    try:
        if len(hole) == 3:
            s = max(pair_strength(list(p), board)
                    for p in itertools.combinations(hole, 2))
        else:
            s = pair_strength(list(hole), board)
    except Exception:
        return 0.5
    return min(0.95, max(0.05, 0.10 + 0.80 * s))


# ------------------------------------------------------------ time governor

SHARE = {"preflop": 0.14, "flop": 0.24, "discard": 0.22,
         "turn": 0.20, "river": 0.20}
QUICK = 0.004   # below this many seconds, skip sampling entirely


def budget_for(kind, req, state):
    """Seconds this decision may spend, from the bank and hands remaining."""
    total = state.get("total_hands") or 200
    hid = state.get("hand_id") or 1
    hands_left = max(1, total - hid + 1)
    bank = float(state.get("time_bank_ms") or 30000)
    left = req.get("time_left_ms")
    if not isinstance(left, (int, float)):
        left = bank
    usable = left / 1000.0 - (0.4 + 0.006 * hands_left)
    if usable <= 0:
        return 0.0
    per_hand = 0.80 * usable / hands_left
    return min(per_hand * SHARE.get(kind, 0.2), 0.5, 0.2 * usable)


# ------------------------------------------------------------ decisions

def choose_discard(req, model, state, rng):
    t0 = perf()
    cards = cards_from_strings(req.get("hole_cards", []))
    flop = cards_from_strings(req.get("board", []))
    if len(cards) != 3 or len(flop) != 3:
        return _safe_reply(req)
    profile = hand_profile(model, state)
    history = state.get("discard_history", [])
    specs = [make_spec(it, profile) for it in build_items(history)]
    budget = budget_for("discard", req, state)
    if budget < QUICK:
        pairs = [[cards[0], cards[1]], [cards[0], cards[2]],
                 [cards[1], cards[2]]]
        values = [pair_strength(p, flop) for p in pairs]
    else:
        pairs, values, _h, _n = flop_equities(
            cards, flop, specs, profile, t0 + budget, rng)
    best = max(values)
    candidates = [i for i, v in enumerate(values) if best - v <= 0.004]
    kept = pairs[rng.choice(candidates)]
    discard = next(c for c in cards if c not in kept)
    state["hole"] = list(kept)
    state["my_discard"] = discard
    return {"type": "discard", "card": _card_string(discard)}


def _predict_fold(model, profile, street, pot, extra, opp_already_raised):
    n = model["street_faced"].get(street, 0)
    base = profile["street_fold"].get(street, 0.5)
    if n < 5:
        base = 0.7 * base + 0.3 * profile["fold"]
    ratio = extra / max(1.0, pot + extra)
    p = base + 0.42 * (ratio - 0.28)
    if opp_already_raised:
        p -= 0.10
    return min(0.82, max(0.08, p))


def _raise_targets(req, pot, my_bet, opp_bet):
    low = req.get("min_raise")
    high = req.get("max_raise")
    if not isinstance(low, int) or not isinstance(high, int) or low > high:
        return []
    to_call = req.get("to_call", 0)
    targets = {low, high}
    if to_call > 0:
        base = pot + to_call
        for f in (0.5, 0.75, 1.0, 1.6):
            targets.add(opp_bet + max(2, int(base * f)))
    else:
        base = max(pot, 2)
        for f in (0.33, 0.5, 0.75, 1.0, 1.5):
            targets.add(my_bet + max(2, int(base * f)))
    return sorted(min(high, max(low, x)) for x in targets)


# Share of the opponent's range lost to folds is the weak part; the hands that
# stay are better than average, so equity when called is lower than raw equity.
ADVERSE_SELECTION = 0.30


def choose_action(req, model, state, rng):
    t0 = perf()
    legal = req.get("legal_actions", [])
    if not isinstance(legal, list):
        return {"type": "action", "action": "fold"}
    if "check" not in legal and "call" not in legal and "fold" not in legal:
        return {"type": "action", "action": "check"}

    board = cards_from_strings(req.get("board", []))
    hole = list(state.get("hole", []))
    if len(board) >= 4 and len(hole) == 3:
        hole = hole[:2]
    street = req.get("street", "preflop")
    profile = hand_profile(model, state)
    history = req.get("history", [])
    specs = [make_spec(it, profile) for it in build_items(history)]
    budget = budget_for(street, req, state)

    equity, samples = 0.5, 1.0
    try:
        if budget < QUICK or len(hole) not in (2, 3):
            equity, samples = quick_equity(hole, board), 15.0
        else:
            deadline = t0 + budget
            if not board:
                equity, samples = preflop_equity(
                    hole, specs, profile, deadline, rng)
            elif len(board) == 3 and len(hole) == 3:
                _p, vals, halves, samples = flop_equities(
                    hole, board, specs, profile, deadline, rng)
                equity = flop_action_equity(vals, halves)
            else:
                equity, samples = equity_post(
                    hole, board, specs, profile, deadline, rng, state)
    except Exception:
        equity, samples = 0.5, 1.0
    equity = min(0.999, max(0.001, equity))
    stderr = math.sqrt(max(0.0, equity * (1.0 - equity)) / max(1.0, samples))
    robust_equity = max(0.0, equity - min(0.06, 0.018 + 0.40 * stderr))

    pot = max(0, int(req.get("pot", 0)))
    to_call = max(0, int(req.get("to_call", 0)))
    my_bet = max(0, int(req.get("my_bet", 0)))
    opp_bet = max(0, int(req.get("opp_bet", 0)))
    my_stack = max(0, int(req.get("my_stack", 0)))
    opp_stack = max(0, int(req.get("opp_stack", 0)))

    call_ev = -1e9
    if "call" in legal:
        all_in_call = to_call >= min(my_stack, opp_stack)
        realization = 1.0 if all_in_call else {
            "preflop": 0.94, "flop": 0.92, "turn": 0.97, "river": 1.0
        }.get(street, 0.92)
        if state.get("seat") == "button" and street in ("flop", "turn", "river"):
            realization = min(1.0, realization + 0.025)
        call_equity = equity if all_in_call else min(0.999, robust_equity * realization)
        call_ev = call_equity * (pot + to_call) - to_call

    best_ev = 0.0
    if "call" in legal:
        best_ev = max(best_ev, call_ev)
    best_raise = None
    if "raise" in legal and equity >= 0.25:
        opp_raised = any(
            x.get("actor") == "opp" and x.get("street") == street
            and x.get("action") == "raise" for x in history
        )
        aggression = profile["aggression"]
        for amount in _raise_targets(req, pot, my_bet, opp_bet):
            extra_self = max(0, amount - my_bet)
            extra_opp = max(0, amount - opp_bet)
            p_fold = _predict_fold(
                model, profile, street, pot, extra_opp, opp_raised)
            reraise_rate = min(0.22, max(0.025, aggression * 0.35))
            if not opp_raised:
                reraise_rate *= 0.65
            p_reraise = (1.0 - p_fold) * reraise_rate
            p_call = (1.0 - p_fold) - p_reraise
            eq_called = max(0.0, equity - ADVERSE_SELECTION * p_fold)
            called_ev = eq_called * (pot + extra_self + extra_opp) - extra_self
            raise_ev = p_fold * pot + p_call * called_ev - p_reraise * extra_self

            threshold = RAISE_EQUITY_THRESHOLDS.get(street, 0.67)
            credible_bluff = p_fold >= 0.64 and equity >= 0.34
            if equity >= threshold or credible_bluff:
                if raise_ev > best_ev + max(0.10, 0.012 * pot):
                    best_ev = raise_ev
                    best_raise = amount

    if best_raise is not None:
        return {"type": "action", "action": "raise", "amount": int(best_raise)}
    if "call" in legal and call_ev > max(0.0, 0.005 * pot):
        return {"type": "action", "action": "call"}
    if "check" in legal:
        return {"type": "action", "action": "check"}
    return {"type": "action", "action": "fold"}



def _record_hand(model, msg, state):
    hand_id = msg.get("hand_id")
    if state.get("last_model_hand") == hand_id:
        return
    history = msg.get("history", [])
    seat = state.get("seat")
    if seat not in ("button", "big_blind"):
        return
    opp_position = "big_blind" if seat == "button" else "button"
    bets = [1, 2] if seat == "button" else [2, 1]
    current_street = "preflop"
    last_aggressor = None
    opp_vpip = False
    opp_pfr = False
    summary = {"pre": 1, "vpip": 0, "pfr": 0, "faced": 0, "folds": 0}

    for item in history:
        street = item.get("street", "preflop")
        if street != current_street:
            current_street = street
            bets = [0, 0]
            last_aggressor = None
        actor_name = item.get("actor")
        actor = 0 if actor_name == "me" else 1
        to_call = max(0, bets[1 - actor] - bets[actor])
        action = item.get("action", "")
        if actor == 1:
            if street == "preflop":
                if action in ("call", "raise"):
                    opp_vpip = True
                if action == "raise":
                    opp_pfr = True
            else:
                model["post_actions"] += 1
                if action == "raise":
                    model["post_raises"] += 1
            if to_call > 0:
                model["faced"] += 1
                model["street_faced"][street] += 1
                summary["faced"] += 1
                if action == "fold" and last_aggressor == 0:
                    model["folds"] += 1
                    model["street_folds"][street] += 1
                    summary["folds"] += 1
        if action in ("call", "raise") and isinstance(item.get("amount"), int):
            bets[actor] = item["amount"]
        if action == "raise":
            last_aggressor = actor

    # VPIP and PFR are hand-level frequencies. Count at most one preflop
    # observation per deal, even when the opponent calls and later faces a
    # reraise. A walk also counts as a hand with no voluntary investment.
    model["pre"] += 1
    model["positions"][opp_position]["pre"] += 1
    if opp_vpip:
        model["vpip"] += 1
        model["positions"][opp_position]["vpip"] += 1
        summary["vpip"] = 1
    if opp_pfr:
        model["pfr"] += 1
        model["positions"][opp_position]["pfr"] += 1
        summary["pfr"] = 1

    if msg.get("result") == "showdown":
        kept = cards_from_strings(msg.get("opp_hole_cards") or [])
        discarded_text = msg.get("opp_discard")
        discarded = CARD_IDS.get(discarded_text) if discarded_text in CARD_IDS else None
        board = cards_from_strings(msg.get("board") or [])
        if len(kept) == 2 and discarded is not None and len(board) == 5:
            original = kept + [discarded]
            model["discard_observations"] += 1
            if simple_discard(original, board[:3]) == discarded:
                model["simple_discard_matches"] += 1
            for item in history:
                if item.get("actor") != "opp":
                    continue
                street = item.get("street", "")
                action = item.get("action", "")
                if street == "preflop":
                    strength = preflop_strength(original)
                elif street == "flop":
                    strength = pair_strength(
                        list(best_pair(original, board[:3])), board[:3]
                    )
                elif street == "turn":
                    strength = pair_strength(kept, board[:4])
                elif street == "river":
                    strength = pair_strength(kept, board[:5])
                else:
                    continue
                key = street + "|" + action
                sample = model["strength_observations"].setdefault(
                    key, [0, 0.0, 0.0]
                )
                sample[0] += 1
                sample[1] += strength
                sample[2] += strength * strength

    model["hands"] += 1
    model["recent"].append(summary)
    if len(model["recent"]) > 30:
        del model["recent"][0]
    state["last_model_hand"] = hand_id


def _history_for_discard(state):
    req = state.get("last_request") or {}
    history = list(req.get("history", []))
    reply = state.get("last_reply")
    if req.get("street") != "flop" or not isinstance(reply, dict):
        return history
    if reply.get("type") != "action":
        return history
    action = reply.get("action")
    if action not in ("check", "call", "raise"):
        return history
    my_entry = {"street": "flop", "actor": "me", "action": action}
    if action == "raise":
        my_entry["amount"] = reply.get("amount")
    elif action == "call":
        my_entry["amount"] = req.get("opp_bet", 0)
    history.append(my_entry)

    if action == "raise":
        # The round can only end here after the opponent calls our bet.
        history.append({
            "street": "flop", "actor": "opp", "action": "call",
            "amount": reply.get("amount", 0),
        })
    elif action == "check":
        opp_checked_before = any(
            x.get("street") == "flop" and x.get("actor") == "opp"
            and x.get("action") == "check" for x in req.get("history", [])
        )
        if not opp_checked_before:
            history.append({"street": "flop", "actor": "opp", "action": "check"})
    return history



STATE = {}
MODEL = empty_model()
RNG = random.Random()


def _lowest(cards):
    return min(cards, key=lambda c: (RANK_VALUE.get(c[:1], 2), SUIT_VALUE.get(c[1:], 0)))


def _safe_reply(msg):
    t = msg.get("type")
    if t == "action_request":
        legal = msg.get("legal_actions", [])
        if not isinstance(legal, list):
            legal = []
        if "check" in legal:
            return {"type": "action", "action": "check"}
        if "call" in legal:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}
    if t == "discard_request":
        cards = msg.get("hole_cards", [])
        if cards:
            return {"type": "discard", "card": _lowest(cards)}
        return {"type": "discard", "card": "2s"}
    return None


def _handle(msg):
    global MODEL, STATE
    t = msg.get("type")
    if t == "init":
        MODEL = empty_model()
        STATE = {
            "time_bank_ms": msg.get("time_bank_ms", 30000),
            "total_hands": msg.get("total_hands", 200),
            "last_model_hand": None,
        }
        return None
    if t == "new_hand":
        seat = msg.get("seat")
        STATE.update({
            "hand_id": msg.get("hand_id"),
            "seat": seat,
            "opp_position": "big_blind" if seat == "button" else "button",
            "hole": cards_from_strings(msg.get("hole_cards", [])),
            "opp_discard": None,
            "my_discard": None,
            "last_request": None,
            "last_reply": None,
            "discard_history": [],
            "hc": {},
            "profile": None,
        })
        return None
    if t == "action_request":
        STATE["last_request"] = msg
        reply = choose_action(msg, MODEL, STATE, RNG)
        STATE["last_reply"] = reply
        return reply
    if t == "discard_request":
        STATE["discard_history"] = _history_for_discard(STATE)
        return choose_discard(msg, MODEL, STATE, RNG)
    if t == "discard_result":
        STATE["opp_discard"] = (
            CARD_IDS.get(msg.get("opp_discard"))
            if msg.get("opp_discard") in CARD_IDS else None
        )
        if msg.get("my_discard") in CARD_IDS:
            STATE["my_discard"] = CARD_IDS[msg.get("my_discard")]
        STATE["hole"] = cards_from_strings(msg.get("hole_cards", []))
        return None
    if t == "hand_result":
        _record_hand(MODEL, msg, STATE)
        return None
    if t == "notice":
        MODEL["notices"] += 1
        return None
    if t == "match_end":
        return "exit"
    return None


def handle(msg):
    """Never raises: any internal error becomes a legal fallback reply."""
    try:
        return _handle(msg)
    except Exception:
        try:
            return _safe_reply(msg)
        except Exception:
            return None


def main():
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(msg, dict):
            continue
        reply = handle(msg)
        if reply == "exit":
            break
        if reply is not None:
            try:
                out.write(json.dumps(reply, separators=(",", ":")) + "\n")
                out.flush()
            except (BrokenPipeError, ValueError, OSError):
                break


if __name__ == "__main__":
    main()