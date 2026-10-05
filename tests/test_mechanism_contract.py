from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from crypto_timing.mechanism_data import aggregate, base_features, dynamic_transform, raw_targets, validate
from crypto_timing.mechanism_model import CausalLevelFilter, LocalEncoder, MechanismConfig, MechanismNetwork, losses
from crypto_timing.mechanism_training import fit_scaler, scaled_labels, gradient_diagnostic
from crypto_timing.mechanism_inference import LevelSignalState, MechanismPredictor


def frame_fixture(n=4000):
    rng = np.random.default_rng(7)
    op = np.exp(np.cumsum(rng.normal(0, .001, n))) * 100
    cl = op * np.exp(rng.normal(0, .0003, n))
    q = rng.uniform(1e4, 2e4, n)
    time = np.arange(n) * 60000 + 1672531200000
    return pd.DataFrame({"date": pd.to_datetime(time, unit="ms", utc=True), "open_time": time,
                         "close_time": time + 59999, "open": op, "close": cl,
                         "high": np.maximum(op, cl) * 1.0002, "low": np.minimum(op, cl) / 1.0002,
                         "volume": q / ((op + cl) / 2), "quote_volume": q, "trade_count": np.ones(n) * 30,
                         "taker_buy_quote_volume": q * .55})


def test_prefix_invariance_and_unit_change():
    frame = frame_fixture()
    short, sigma, _ = base_features(frame.iloc[:3000], 60)
    full, fs, _ = base_features(frame, 60)
    np.testing.assert_allclose(short, full[:3000], equal_nan=True)
    np.testing.assert_allclose(sigma, fs[:3000], equal_nan=True)
    changed = frame.copy()
    for col in ("quote_volume", "taker_buy_quote_volume", "volume"):
        changed[col] *= 1000
    cx, cs, _ = base_features(changed, 60)
    np.testing.assert_allclose(short[:, 6], cx[:3000, 6], equal_nan=True, atol=1e-10)
    np.testing.assert_allclose(short[:, 8], cx[:3000, 8], equal_nan=True)
    np.testing.assert_allclose(dynamic_transform(short, sigma)[:, :10], dynamic_transform(cx, cs)[:3000, :10], equal_nan=True, atol=1e-9)


def test_current_jump_does_not_scale_its_own_denominator():
    frame = frame_fixture()
    _, sigma, _ = base_features(frame, 60)
    frame.loc[3000, "close"] *= 2
    _, altered, _ = base_features(frame, 60)
    assert altered[3000] == sigma[3000]
    assert altered[3001] > sigma[3001]


def test_shared_entry_expiry_and_semivariance_definition():
    frame = frame_fixture(6000)
    labels = raw_targets(frame, np.full(100, .01))
    e = 65
    assert labels[0, 1] == pytest.approx(frame.open.iloc[e+240]/frame.open.iloc[e]-1, rel=1e-6)
    for sampling, column in ((1, 5), (5, 3)):
        prices = frame.open.to_numpy()[e:e+241:sampling]
        returns = np.diff(np.log(prices))
        assert returns.sum() == pytest.approx(np.log1p(labels[0, 1]), abs=1e-8)
        assert labels[0, column] == pytest.approx((np.maximum(returns, 0)**2).sum(), rel=1e-6)
        assert labels[0, column+1] == pytest.approx((np.minimum(returns, 0)**2).sum(), rel=1e-6)
    assert np.isnan(labels[-1, 1])


def test_aggregation_completion_and_missing_semantics():
    frame = frame_fixture(600)
    five = aggregate(frame, 5)
    assert five.close_time.iloc[0] == frame.close_time.iloc[4]
    assert five.quote_volume.iloc[0] == pytest.approx(frame.quote_volume.iloc[:5].sum())
    validate(five, 5)
    with pytest.raises(ValueError):
        validate(frame.drop(index=1))
    frame.loc[3, "quote_volume"] = 0
    frame.loc[3, "taker_buy_quote_volume"] = 0
    features, _, _ = base_features(frame, 60)
    assert np.isnan(features[3, 8]) and features[3, 18] == 1


def test_train_only_scaler_and_bounded_zero():
    x = np.random.default_rng(1).normal(size=(100, 2, 3))
    a, b = fit_scaler(x, 50, (1,))
    x[50:] *= 1e6
    c, d = fit_scaler(x, 50, (1,))
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert (a[:, 1] == 0).all() and (b[:, 1] == 1).all()


def test_scaled_label_objects_remain_distinct():
    raw = np.ones((2, 1, 9), np.float32) * .01
    raw[..., 3], raw[..., 4] = .0003, .0001
    raw[..., 5], raw[..., 6] = .0001, .0001
    five, scale = scaled_labels(raw, np.array([.02]))
    one, _ = scaled_labels(raw, np.array([.02]), 1)
    assert scale[0, 0] == pytest.approx(.02)
    assert five[0, 0, 4] == pytest.approx(.5)
    assert one[0, 0, 4] == 0
    assert five[0, 0, 5] < 1


def test_causal_encoder_and_level_filter():
    torch.manual_seed(1)
    model = LocalEncoder(3, 12).eval()
    x = torch.randn(2, 30, 3)
    mask = torch.ones_like(x)
    original = model(x, mask)
    x[:, 20:] += 100
    torch.testing.assert_close(original[:, :20], model(x, mask)[:, :20])
    filt = CausalLevelFilter()
    v = torch.ones(60, 2) * 3
    torch.testing.assert_close(filt(v), v)
    v[40:] *= 2
    torch.testing.assert_close(filt(v)[:40], torch.ones(40, 2)*3)
    filt(v).square().mean().backward()
    assert filt.logit.grad is not None


@pytest.mark.parametrize("sharing", ["partial", "isolated", "full"])
def test_sharing_gradient_paths(sharing):
    cfg = MechanismConfig(sharing=sharing, width=64)
    net = MechanismNetwork(cfg).eval()
    args = (torch.randn(2,144,14), torch.ones(2,144,14), torch.randn(2,168,10),
            torch.ones(2,168,10), torch.randn(2,24), torch.ones(2,24))
    output = net(*args)
    y = torch.zeros(2, 8)
    primary, auxiliary = losses(output, y, cfg)
    diagnostic = gradient_diagnostic(primary, auxiliary, net.shared_parameters())
    assert all(np.isfinite(diagnostic))
    grads = torch.autograd.grad(auxiliary, net.fast.input[0].weight, allow_unused=True, retain_graph=True)[0]
    assert (grads is None) == (sharing == "isolated")
    memory_grad = torch.autograd.grad(auxiliary, net.memory.project[0].weight, allow_unused=True, retain_graph=True)[0]
    assert (memory_grad is not None) == (sharing == "full")
    primary.backward()
    assert net.fast.input[0].weight.grad.abs().sum() > 0


def test_distribution_readout_does_not_use_future_routing():
    cfg = MechanismConfig(label="four_bin", width=64)
    net = MechanismNetwork(cfg).eval()
    args = (torch.randn(2,144,14), torch.ones(2,144,14), torch.randn(2,168,10),
            torch.ones(2,168,10), torch.randn(2,24), torch.ones(2,24), None, None,
            torch.tensor([-.2,-.7,-1.5,-3.,.2,.7,1.5,3.]))
    out = net(*args)
    expect = (out["band"].softmax(-1) * (out["sign"].sigmoid()*args[-1][4:]
              +(1-out["sign"].sigmoid())*args[-1][:4])).sum(-1)
    torch.testing.assert_close(out["signal"], expect)


def test_pure_path_trains_all_representation_branches():
    net = MechanismNetwork(MechanismConfig(label="path", memory="gru"))
    args = (torch.randn(2,144,14), torch.ones(2,144,14), torch.randn(2,168,10),
            torch.ones(2,168,10), torch.randn(2,24), torch.ones(2,24))
    out = net(*args)
    out["path"].square().mean().backward()
    assert net.memory.project[0].weight.grad.abs().sum() > 0
    assert net.slow.input[0].weight.grad.abs().sum() > 0


def test_level_state_is_not_impulse_accumulation():
    state=LevelSignalState(2)
    for h in range(10):
        np.testing.assert_array_equal(state.update(np.datetime64("2026-01-01")+np.timedelta64(h,"h"),np.array([.2,-.3])),[.2,-.3])
    with pytest.raises(ValueError):
        state.update("2026-01-01T09:00:00",np.array([.2,-.3]))


def test_checkpoint_inference_has_no_future_label_dependency(tmp_path):
    cfg=MechanismConfig(label="return",width=64,memory="mean")
    net=MechanismNetwork(cfg)
    scaler={f"dynamic_{name}":{"center":np.zeros((2,dim)).tolist(),"scale":np.ones((2,dim)).tolist()}
            for name,dim in (("fast",20),("slow",10),("state",24))}
    checkpoint=tmp_path/"model.pt"
    torch.save({"config":cfg.__dict__,"model":net.state_dict(),"scalers":scaler,"scale_floors":[.01,.01],
                "representatives":[-3,-1.5,-.7,-.2,.2,.7,1.5,3]},checkpoint)
    predictor=MechanismPredictor(checkpoint,device="cpu")
    x={"fast":np.zeros((2,144,20)),"slow":np.zeros((2,168,10)),"state":np.zeros((2,24))}
    out=predictor.predict(x,np.array([0,1]),np.array([.02,.02]))
    assert np.isfinite(out["signal_level"]).all()
    np.testing.assert_allclose(out["return_estimate"],out["signal_level"]*.02)
    x["fast"][...,-1]=1e6  # diagnostic return copy is excluded from inference too
    other=predictor.predict(x,np.array([0,1]),np.array([.02,.02]))
    np.testing.assert_array_equal(out["signal_level"],other["signal_level"])


def test_four_bin_zero_return_is_half_direction_target():
    cfg=MechanismConfig(label="four_bin")
    sign=torch.zeros(2,4,requires_grad=True)
    output={"path":torch.zeros(2,3),"band":torch.zeros(2,4,requires_grad=True),"sign":sign}
    primary,_=losses(output,torch.zeros(2,8),cfg)
    primary.backward()
    torch.testing.assert_close(sign.grad,torch.zeros_like(sign))
