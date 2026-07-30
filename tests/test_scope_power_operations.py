"""Regression tests for channel arithmetic and power-energy calculations."""

import numpy as np
import numpy.testing as npt
import pytest

import pysignalscope as pss
from pysignalscope.scope_dataclass import Channel


def _channel(time, data, *, label=None, unit=None):
    """Create a valid test channel."""
    return pss.Scope.generate_channel(
        time=np.asarray(time, dtype=float),
        data=np.asarray(data, dtype=float),
        label=label,
        unit=unit,
    )


def test_multiply_calculates_power_and_generates_label():
    """Voltage and current samples on the same time base produce power."""
    voltage = _channel([0.0, 1.0, 2.0], [2.0, 3.0, 4.0], label="Voltage", unit="V")
    current = _channel([0.0, 1.0, 2.0], [5.0, 6.0, 7.0], label="Current", unit="A")

    power = pss.Scope.multiply(voltage, current)

    npt.assert_allclose(power.data, [10.0, 18.0, 28.0])
    npt.assert_array_equal(power.time, voltage.time)
    assert power.label == "Voltage * Current"
    assert power.unit == "W"


@pytest.mark.parametrize(
    "second_time",
    [
        [0.0, 1.0],
        [0.0, 1.1, 2.0],
    ],
)
def test_multiply_rejects_mismatched_timebases(second_time):
    """Point-wise multiplication must not combine differently sampled channels."""
    voltage = _channel([0.0, 1.0, 2.0], [2.0, 3.0, 4.0])
    current = _channel(second_time, np.ones(len(second_time)))

    with pytest.raises(ValueError, match="Channel time bases"):
        pss.Scope.multiply(voltage, current)


def test_multiply_rejects_malformed_channel_shape():
    """Directly constructed malformed channels are rejected before broadcasting."""
    voltage = Channel(
        time=np.array([0.0, 1.0, 2.0]),
        data=np.array([2.0, 3.0]),
        label=None,
        unit="V",
        color=None,
        linestyle=None,
        source=None,
        modulename="scope",
    )
    current = _channel([0.0, 1.0, 2.0], [1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="must have the same shape"):
        pss.Scope.multiply(voltage, current)


def test_integrate_uses_actual_nonuniform_time_steps():
    """Cumulative energy is correct for non-equidistant scope samples."""
    power = _channel([0.0, 0.5, 2.0], [0.0, 2.0, 2.0], unit="W")

    energy = pss.Scope.integrate(power)

    npt.assert_allclose(energy.data, [0.0, 0.5, 3.5])
    assert energy.label == "Energy"
    assert energy.unit == "J"


def test_integrate_accepts_two_samples_and_custom_label():
    """A two-sample waveform has one valid trapezoidal interval."""
    power = _channel([0.0, 2.0], [3.0, 3.0], unit="W")

    energy = pss.Scope.integrate(power, label="Turn-on energy")

    npt.assert_allclose(energy.data, [0.0, 6.0])
    assert energy.label == "Turn-on energy"


def test_integrate_rejects_single_sample_channel():
    """Integration requires at least one time interval."""
    power = _channel([0.0], [5.0], unit="W")

    with pytest.raises(ValueError, match="At least two channel samples"):
        pss.Scope.integrate(power)


def test_integrate_rejects_non_string_label():
    """The optional label accepts only strings or None."""
    power = _channel([0.0, 1.0], [1.0, 1.0], unit="W")

    with pytest.raises(TypeError, match="str or None"):
        pss.Scope.integrate(power, label=123)


@pytest.mark.parametrize("operation", [pss.Scope.add, pss.Scope.subtract])
def test_add_and_subtract_reject_mismatched_timebases(operation):
    """All element-wise arithmetic operations share the same safety checks."""
    channel_1 = _channel([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    channel_2 = _channel([0.0, 1.1, 2.0], [4.0, 5.0, 6.0])

    with pytest.raises(ValueError, match="identical values"):
        operation(channel_1, channel_2)
