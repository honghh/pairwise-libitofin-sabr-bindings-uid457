"""Backward-flat interpolation on the SABR swaption vol cube facade (#606).

The oracle is the core's QuantLib 1.43 fixture
(``libitofin/tests/fixtures/sabr_backward_flat/oracle.csv``): the CommonVars
cube - the moving 6x4 ATM matrix, the 3x3 cube with five strike spreads,
guess [0.2, 0.5, 0.4, 0.0], every parameter free - queried at option tenors
between the parameter-cube nodes, once sparse and once ATM-calibrated, with
backward_flat off and on. Every served vol must match C++ within 1e-6, and
off the dense nodes the two flags disagree by more than 1e-3, so the samples
catch a silently ignored flag, a transposed axis, or a wrong dense
interpolation.

Beyond the oracle, this file pins the facade contract: an omitted
backward_flat is the explicit False, a True build keeps its backward-flat
behaviour across quote bumps and evaluation-date moves, and the interpolated
cube rejects the flag instead of ignoring it - after which the session keeps
working.
"""

# standard library
import csv
from pathlib import Path

# pypi/conda library
import pytest

# itofin library
from itofin import ItofinError, Settings
from itofin.indexes import Currency, Euribor, SwapIndex
from itofin.quotes import SimpleQuote
from itofin.termstructures import (
    FlatForward,
    InterpolatedSwaptionVolatilityCube,
    SabrSwaptionVolatilityCube,
    SwaptionVolatilityMatrix,
    VolatilityType,
)
from itofin.time import BusinessDayConvention, Calendar, Date, DayCounter, Period

ORACLE = (
    Path(__file__).resolve().parents[2]
    / "libitofin/tests/fixtures/sabr_backward_flat/oracle.csv"
)

EVAL = Date(15, 6, 2026)
BDC = BusinessDayConvention.ModifiedFollowing

ATM_OPTION_TENORS = [
    Period(1, "Months"),
    Period(6, "Months"),
    Period(1, "Years"),
    Period(5, "Years"),
    Period(10, "Years"),
    Period(30, "Years"),
]
ATM_SWAP_TENORS = [
    Period(1, "Years"),
    Period(5, "Years"),
    Period(10, "Years"),
    Period(30, "Years"),
]
ATM_VOLS = [
    [0.1300, 0.1560, 0.1390, 0.1220],
    [0.1440, 0.1580, 0.1460, 0.1260],
    [0.1600, 0.1590, 0.1470, 0.1290],
    [0.1640, 0.1470, 0.1370, 0.1220],
    [0.1400, 0.1300, 0.1250, 0.1100],
    [0.1130, 0.1090, 0.1070, 0.0930],
]
OPTION_TENORS = [Period(1, "Years"), Period(10, "Years"), Period(30, "Years")]
SWAP_TENORS = [Period(2, "Years"), Period(10, "Years"), Period(30, "Years")]
STRIKE_SPREADS = [-0.020, -0.005, 0.000, 0.005, 0.020]
VOL_SPREADS = [
    [0.0599, 0.0049, 0.0000, -0.0001, 0.0127],
    [0.0729, 0.0086, 0.0000, -0.0024, 0.0098],
    [0.0738, 0.0102, 0.0000, -0.0039, 0.0065],
    [0.0465, 0.0063, 0.0000, -0.0032, -0.0010],
    [0.0558, 0.0084, 0.0000, -0.0050, -0.0057],
    [0.0576, 0.0083, 0.0000, -0.0043, -0.0014],
    [0.0437, 0.0059, 0.0000, -0.0030, -0.0006],
    [0.0533, 0.0078, 0.0000, -0.0045, -0.0046],
    [0.0545, 0.0079, 0.0000, -0.0042, -0.0020],
]
PARAMETERS_GUESS = [0.2, 0.5, 0.4, 0.0]

SETTINGS = Settings()
SETTINGS.set_evaluation_date(EVAL)


def _swap_index(tenor, ibor_index):
    """An EuriborSwapIsdaFixA-convention index, as in the core fixture."""
    return SwapIndex(
        "EuriborSwapIsdaFixA",
        tenor,
        2,
        Currency.eur(),
        Calendar.target(),
        Period(1, "Years"),
        BDC,
        DayCounter.thirty360_bond_basis(),
        ibor_index,
        SETTINGS,
    )


def _atm_matrix():
    """The moving 6x4 ATM surface of the CommonVars fixture."""
    return SwaptionVolatilityMatrix.moving(
        Calendar.target(),
        BDC,
        ATM_OPTION_TENORS,
        ATM_SWAP_TENORS,
        [[SimpleQuote(vol) for vol in row] for row in ATM_VOLS],
        DayCounter.actual365_fixed(),
        VolatilityType.ShiftedLognormal,
        SETTINGS,
    )


def _build_cube(is_atm_calibrated, backward_flat, live_spread=None):
    """The CommonVars SABR cube. ``backward_flat=None`` omits the keyword
    entirely; ``live_spread`` substitutes a live quote at the (10Y, 2Y)
    node's first strike spread so bumps can be observed."""
    curve = FlatForward(EVAL, 0.05, DayCounter.actual360())
    euribor6m = Euribor.six_months(curve, SETTINGS)
    spreads = [[SimpleQuote(spread) for spread in row] for row in VOL_SPREADS]
    if live_spread is not None:
        spreads[3][0] = live_spread
    kwargs = {}
    if backward_flat is not None:
        kwargs["backward_flat"] = backward_flat
    return SabrSwaptionVolatilityCube(
        _atm_matrix(),
        OPTION_TENORS,
        SWAP_TENORS,
        STRIKE_SPREADS,
        spreads,
        _swap_index(Period(2, "Years"), euribor6m),
        _swap_index(Period(1, "Years"), euribor6m),
        [[SimpleQuote(guess) for guess in PARAMETERS_GUESS] for _ in VOL_SPREADS],
        [False, False, False, False],
        is_atm_calibrated,
        SETTINGS,
        **kwargs,
    )


@pytest.fixture(scope="module")
def oracle_cubes():
    """The four oracle cubes: sparse/ATM-calibrated x backward_flat off/on."""
    return {
        (arm, flag): _build_cube(bool(arm), bool(flag))
        for arm in (0, 1)
        for flag in (0, 1)
    }


def test_backward_flat_matches_the_quantlib_oracle(oracle_cubes):
    rows = 0
    with ORACLE.open() as stream:
        for row in csv.DictReader(stream):
            arm = int(row["is_atm_calibrated"])
            option = Period(int(row["option_tenor"].rstrip("Y")), "Years")
            swap = Period(int(row["swap_tenor"].rstrip("Y")), "Years")
            strike = float(row["strike"])
            served = {}
            for flag, column in ((0, "vol_bilinear"), (1, "vol_backward_flat")):
                expected = float(row[column])
                got = oracle_cubes[(arm, flag)].volatility(option, swap, strike, True)
                assert abs(got - expected) < 1e-6, (
                    f"arm {arm} backward_flat {flag} "
                    f"{row['option_tenor']}x{row['swap_tenor']}: "
                    f"got {got}, C++ {expected}"
                )
                served[flag] = got
            on_dense_node = arm == 1 and row["option_tenor"] == "5Y"
            gap = abs(served[0] - served[1])
            if on_dense_node:
                assert gap < 1e-12, "5Y is a dense node once ATM-calibrated"
            else:
                assert gap > 1e-3, "the sample must detect an ignored backward_flat"
            rows += 1
    assert rows == 72


def test_an_omitted_backward_flat_is_the_explicit_false():
    coordinates = (Period(2, "Years"), Period(5, "Years"))
    for is_atm_calibrated in (False, True):
        omitted = _build_cube(is_atm_calibrated, None)
        explicit = _build_cube(is_atm_calibrated, False)
        strike = omitted.atm_strike_from_tenor(*coordinates)
        served_omitted = omitted.volatility(*coordinates, strike, True)
        served_explicit = explicit.volatility(*coordinates, strike, True)
        assert abs(served_omitted - served_explicit) < 1e-14


def test_backward_flat_survives_quote_bumps_and_evaluation_date_moves():
    coordinates = (Period(2, "Years"), Period(5, "Years"))
    strike = 0.05
    for is_atm_calibrated in (False, True):
        live = SimpleQuote(VOL_SPREADS[3][0])
        cube = _build_cube(is_atm_calibrated, True, live_spread=live)
        initial = cube.volatility(*coordinates, strike, True)
        bilinear = _build_cube(is_atm_calibrated, False, live_spread=live)
        assert abs(initial - bilinear.volatility(*coordinates, strike, True)) > 1e-3

        live.set_value(VOL_SPREADS[3][0] + 0.01)
        after_quote = cube.volatility(*coordinates, strike, True)
        assert abs(after_quote - initial) > 1e-7
        rebuilt = _build_cube(is_atm_calibrated, True, live_spread=live)
        assert abs(after_quote - rebuilt.volatility(*coordinates, strike, True)) < 1e-14
        bilinear = _build_cube(is_atm_calibrated, False, live_spread=live)
        assert abs(after_quote - bilinear.volatility(*coordinates, strike, True)) > 1e-3

        moved = Calendar.target().advance(EVAL, 1, "Days", BDC, False)
        SETTINGS.set_evaluation_date(moved)
        try:
            after_move = cube.volatility(*coordinates, strike, True)
            assert abs(after_move - after_quote) > 1e-10
            rebuilt = _build_cube(is_atm_calibrated, True, live_spread=live)
            assert abs(after_move - rebuilt.volatility(*coordinates, strike, True)) < 1e-14
            bilinear = _build_cube(is_atm_calibrated, False, live_spread=live)
            assert abs(after_move - bilinear.volatility(*coordinates, strike, True)) > 1e-3
        finally:
            SETTINGS.set_evaluation_date(EVAL)


def test_the_interpolated_cube_rejects_backward_flat_and_the_session_survives():
    curve = FlatForward(EVAL, 0.05, DayCounter.actual360())
    euribor6m = Euribor.six_months(curve, SETTINGS)
    atm = _atm_matrix()
    spreads = [[SimpleQuote(spread) for spread in row] for row in VOL_SPREADS]
    bases = (
        _swap_index(Period(2, "Years"), euribor6m),
        _swap_index(Period(1, "Years"), euribor6m),
    )
    with pytest.raises(ItofinError, match="backward_flat"):
        InterpolatedSwaptionVolatilityCube(
            atm,
            OPTION_TENORS,
            SWAP_TENORS,
            STRIKE_SPREADS,
            spreads,
            *bases,
            SETTINGS,
            backward_flat=True,
        )
    cube = InterpolatedSwaptionVolatilityCube(
        atm,
        OPTION_TENORS,
        SWAP_TENORS,
        STRIKE_SPREADS,
        spreads,
        *bases,
        SETTINGS,
    )
    strike = cube.atm_strike_from_tenor(Period(1, "Years"), Period(2, "Years"))
    assert cube.volatility(Period(1, "Years"), Period(2, "Years"), strike, True) > 0.0
