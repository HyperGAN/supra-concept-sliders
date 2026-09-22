import pytest

from supra.convergence import Plateau


def test_requires_plateau_at_reduced_lr_and_keeps_best():
    p = Plateau(lr=4e-5, min_lr=1e-5, patience=2, final_patience=3, min_step=0)
    assert p.observe(400, 0.5) == ("continue", True)
    assert p.observe(800, 0.4) == ("continue", True)
    assert p.observe(1200, 0.401)[0] == "continue"
    assert p.observe(1600, 0.402)[0] == "reduce_lr"
    assert p.lr == 2e-5
    assert p.observe(2000, 0.399)[0] == "continue"
    assert p.observe(2400, 0.401)[0] == "continue"
    assert p.observe(2800, 0.403)[0] == "reduce_lr"
    assert p.lr == 1e-5
    # Better minimum resets the stage even at the floor.
    assert p.observe(3200, 0.39) == ("continue", True)
    assert p.observe(3600, 0.40)[0] == "continue"
    assert p.observe(4000, 0.40)[0] == "continue"
    assert p.observe(4400, 0.40)[0] == "converged"
    assert p.best_step == 3200


def test_accumulated_small_improvements_reset_patience():
    p = Plateau(patience=4, min_step=0)
    p.observe(400, 1)
    assert p.observe(800, 0.998)[0] == "continue"
    assert p.stale == 1
    p.observe(1200, 0.996)
    p.observe(1600, 0.994)
    assert p.stale == 0
    assert p.best == 0.994
    with pytest.raises(ValueError):
        p.observe(2000, float("nan"))
