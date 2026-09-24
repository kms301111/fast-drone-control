"""표6 사다리의 S1(시간 정렬)과 C2(첫 입력 배분값 근처 제한) 검증.

논문 표6의 제거실험 사다리는 세 축으로 이루어진다.
  V13-0 → V13-1 : A0 → A1  (총추력 우선 배분)        → test_allocation_a1.py
  V13-1 → V13-2 : C0 → C1  (배분 결과 피드백)         → test_allocation_a1.py
  V13-2 → V13   : S0 → S1  (시간 정렬, 식22-23)       → 이 파일
  V13   → V13-C2: 연결 강도 (5.4절 변형)              → 이 파일

두 플래그 모두 기본값이 '기존 동작'이어야 한다. 기존 실험 결과(results/)가
플래그 추가만으로 바뀌면 사다리의 어느 칸이 원인인지 못 가리기 때문이다.
"""
import numpy as np
import pytest

from control.vehicle_params import vehicle_params as P
from control.hybrid_comparison import ProperHybrid, VirtualNMPC


class _StubNMPC:
    """결정적 가상명령. IPOPT를 빼고 INDI 측정 경로만 시험한다."""

    def __init__(self, params):
        self.mg = params['mass'] * params['g']

    def __call__(self, t, x):
        return np.array([self.mg, 0.0, 0.0, 0.0])

    def set_prev_input(self, v):
        pass


def _rotor_thrust(params, n, V_axial):
    """ProperHybrid 안의 전진비 보정 추력식을 독립적으로 재계산한다.

    같은 함수를 부르면 서로를 검증하지 못하므로 손으로 다시 쓴다.
    """
    n = np.asarray(n, dtype=float)
    n_rps = n / (2 * np.pi)
    J = V_axial / (n_rps * params['D_prop'] + 1e-8)
    fac = np.maximum(1.0 - J / params['J_max'], 0.0)
    return params['k_T'] * n**2 * fac


def _state(n, v=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0), z=50.0):
    x = np.zeros(17)
    x[2] = z
    x[3:6] = v
    x[6:10] = [0.0, 0.0, 0.0, 1.0]      # z축 호버 (scalar-last)
    x[10:13] = omega
    x[13:17] = n
    return x


# ── S0/S1 ────────────────────────────────────────────────────────────

def test_s0_is_the_default():
    """플래그를 안 주면 기존 동작이어야 한다."""
    h = ProperHybrid(_StubNMPC(P), P)
    assert h.time_align == 'S0'
    assert h._f_prev is None and h._f_filt is None   # S1 상태는 만들지도 않음


def test_bad_time_align_is_rejected():
    with pytest.raises(ValueError, match='time_align'):
        ProperHybrid(_StubNMPC(P), P, time_align='S2')


def test_s1_lpf_coefficient_follows_eq23():
    """식(23) λ = 1 − exp(−2π f_c Δt). S0의 후향차분 근사와 구별된다."""
    dt, f_cut = 1e-3, 50.0
    s1 = ProperHybrid(_StubNMPC(P), P, dt=dt, f_cut=f_cut, time_align='S1')
    s0 = ProperHybrid(_StubNMPC(P), P, dt=dt, f_cut=f_cut, time_align='S0')

    assert s1._lpf_coeff(dt) == pytest.approx(1.0 - np.exp(-2*np.pi*f_cut*dt))
    assert s0._lpf_coeff(dt) == pytest.approx(dt / (dt + 1.0/(2*np.pi*f_cut)))
    # 같은 필터의 다른 이산화 — 가깝지만 같지 않다(1 ms/50 Hz에서 약 12%).
    assert s0._lpf_coeff(dt) < s1._lpf_coeff(dt)
    assert s1._lpf_coeff(dt) / s0._lpf_coeff(dt) == pytest.approx(1.126, abs=2e-3)


def test_s1_applies_mid_sample_alignment_and_the_same_filter():
    """식(22) f_mid=(f_k+f_{k-1})/2 와 식(23) 필터를 손계산과 대조한다."""
    dt = 1e-3
    h = ProperHybrid(_StubNMPC(P), P, dt=dt, time_align='S1')
    n_eq = np.sqrt(P['mass']*P['g'] / (4*P['k_T']))
    seq = [n_eq*np.ones(4), n_eq*np.array([1.05, 1.0, .95, 1.0]),
           n_eq*np.array([1.10, 1.0, .90, 1.0])]

    # 1번째 호출은 초기화(_fallback)라 S1 상태를 건드리지 않는다.
    h(0.0, _state(seq[0]))
    assert h._f_prev is None

    # 2번째: f_{k-1}이 없으니 f_mid=f_k, 필터도 그 값으로 초기화된다.
    h(dt, _state(seq[1]))
    f2 = _rotor_thrust(P, seq[1], 0.0)
    np.testing.assert_allclose(h._f_prev, f2, rtol=1e-12)
    np.testing.assert_allclose(h._f_filt, f2, rtol=1e-12)

    # 3번째: 중간값 정렬 + 각가속도와 같은 계수의 LPF.
    h(2*dt, _state(seq[2]))
    f3 = _rotor_thrust(P, seq[2], 0.0)
    alpha = 1.0 - np.exp(-2*np.pi*50.0*dt)
    expected = alpha*0.5*(f3 + f2) + (1 - alpha)*f2
    np.testing.assert_allclose(h._f_filt, expected, rtol=1e-12)


def test_s1_thrust_filter_starts_at_hover_not_zero():
    """LPF를 0에서 시작하면 INDI가 '없는 추력 부족'을 크게 본다."""
    h = ProperHybrid(_StubNMPC(P), P, time_align='S1')
    n_eq = np.sqrt(P['mass']*P['g'] / (4*P['k_T']))
    h(0.0, _state(n_eq*np.ones(4)))
    h(1e-3, _state(n_eq*np.ones(4)))
    # 첫 유효 표본에서 이미 호버 추력 근처여야 한다(0에서 올라오지 않는다).
    assert h._f_filt.sum() == pytest.approx(P['mass']*P['g'], rel=1e-6)


def test_s1_matches_s0_on_smooth_input_and_diverges_on_noise():
    """S1의 효과는 측정 잡음에 비례한다 — 논문 4.5절의 '가상 외란' 주장."""
    n_eq = np.sqrt(P['mass']*P['g'] / (4*P['k_T']))

    def run(align, jitter, seed=3):
        h = ProperHybrid(_StubNMPC(P), P, dt=1e-3, alloc_mode='A1',
                         time_align=align)
        rng = np.random.default_rng(seed)
        out = []
        for k in range(600):
            t = k*1e-3
            n = n_eq*(1 + .05*np.sin(2*np.pi*2*t + np.arange(4)))
            if jitter:
                n = n*(1 + jitter*rng.standard_normal(4))
            out.append(h(t, _state(n, v=(12., 0., 0.),
                                   omega=(.2*np.sin(4*t), 0., 0.))))
        return np.array(out)

    smooth = np.abs(run('S0', 0.0) - run('S1', 0.0)).max()
    noisy = np.abs(run('S0', 0.02) - run('S1', 0.02)).max()
    # 매끄러운 신호에서는 위상 보정 수준(회전수 상한의 1% 미만).
    assert smooth < 0.01*P['n_max']
    # 잡음이 있으면 뚜렷하게 갈린다(필터가 지터를 걸러내므로).
    assert noisy > 5*smooth


def test_reset_clears_s1_state():
    """MC 시행 간 독립성 — 남은 필터 상태가 다음 시행에 새면 안 된다."""
    h = ProperHybrid(_StubNMPC(P), P, time_align='S1')
    n_eq = np.sqrt(P['mass']*P['g'] / (4*P['k_T']))
    h(0.0, _state(n_eq*np.ones(4)))
    h(1e-3, _state(n_eq*np.ones(4)))
    assert h._f_filt is not None
    h.reset()
    assert h._f_prev is None and h._f_filt is None


# ── C0/C1/C2 ─────────────────────────────────────────────────────────

def test_c2_is_off_by_default():
    """기본값에서는 제약이 추가되지 않아야 한다(g 차원 불변)."""
    base = VirtualNMPC(P, v_ref=[15, 0, 0], z_ref=50.)
    assert base.c2_limit is None
    with_c1 = VirtualNMPC(P, v_ref=[15, 0, 0], z_ref=50., alloc_feedback=True)
    assert with_c1.lbg.size == base.lbg.size


def test_c2_requires_c1_and_a_positive_limit():
    """C1 없이 C2를 켜면 '트림 주변 제한'이 되어 다른 실험이 된다."""
    with pytest.raises(ValueError, match='alloc_feedback'):
        VirtualNMPC(P, c2_limit=0.5)
    with pytest.raises(ValueError, match='positive'):
        VirtualNMPC(P, c2_limit=0.0, alloc_feedback=True)


@pytest.mark.parametrize('limit', [0.30, 0.05])
def test_c2_constrains_the_first_input_near_the_allocation(limit):
    """‖Dν⁻¹(ν_0 − ν_alloc)‖ ≤ limit 이 실제로 구속되는지.

    냉시동은 기본 max_iter=30으로 수렴하지 못해 근사 실행가능해에 머문다.
    여기서는 제약식 자체를 보려는 것이므로 반복 상한을 넉넉히 준다.
    """
    mg = P['mass']*P['g']
    D_nu = np.array([mg, 100., 100., 100.])
    nu_alloc = np.array([0.70*mg, 0., 0., 0.])   # NMPC가 원하는 값에서 멀리
    x = _state(np.sqrt(mg/(4*P['k_T']))*np.ones(4), v=(12., 0., 0.))

    free = VirtualNMPC(P, v_ref=[15, 0, 0], z_ref=50.,
                       alloc_feedback=True, max_iter=300)
    free.set_prev_input(nu_alloc)
    d_free = np.linalg.norm((free(0., x) - nu_alloc) / D_nu)

    c2 = VirtualNMPC(P, v_ref=[15, 0, 0], z_ref=50., alloc_feedback=True,
                     c2_limit=limit, max_iter=300)
    c2.set_prev_input(nu_alloc)
    d_c2 = np.linalg.norm((c2(0., x) - nu_alloc) / D_nu)

    assert c2.lbg.size == free.lbg.size + 1      # 제약 한 줄 추가
    assert d_free > limit                        # 제한이 의미 있는 상황인지 먼저
    assert d_c2 <= limit + 1e-6                  # 만족
    assert d_c2 == pytest.approx(limit, rel=1e-3)  # 활성 제약이라 경계에 붙는다
