"""Unit tests for the Elo rating module (synthetic, hand-computable data)."""

import pytest

from nfl_predictor.predictor.elo import derive_home_field_advantage


# =============================================================================
# PART 1: home-field advantage derivation
#   HFA = -400 * log10((1 / home_win_rate) - 1)
# =============================================================================


def test_hfa_rate_ten_elevenths_is_400():
    # 1 / (10/11) = 1.1;  1.1 - 1 = 0.1;  log10(0.1) = -1;  -400 * -1 = 400
    assert derive_home_field_advantage(10 / 11) == pytest.approx(400.0)


def test_hfa_rate_one_eleventh_is_minus_400():
    # 1 / (1/11) = 11;  11 - 1 = 10;  log10(10) = 1;  -400 * 1 = -400
    assert derive_home_field_advantage(1 / 11) == pytest.approx(-400.0)


def test_hfa_rate_sixty_percent():
    # 1 / 0.6 = 5/3;  5/3 - 1 = 2/3;  log10(2/3) = log10(2) - log10(3)
    #   = 0.30103 - 0.47712 = -0.17609;  -400 * -0.17609 = 70.4365
    assert derive_home_field_advantage(0.6) == pytest.approx(70.4365, abs=1e-4)


def test_hfa_coin_flip_is_exactly_zero():
    # 1 / 0.5 = 2;  2 - 1 = 1;  log10(1) = 0;  -400 * 0 = 0
    assert derive_home_field_advantage(0.5) == 0.0


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1, 1.5])
def test_hfa_rejects_rates_outside_open_unit_interval(rate):
    with pytest.raises(ValueError):
        derive_home_field_advantage(rate)
