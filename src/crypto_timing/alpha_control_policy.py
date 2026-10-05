"""Frozen forecasts and friction-aware portfolio control; no model fitting.

The convex *unconstrained* objective is solved by proximal gradient. Portfolio
limits are subsequently enforced by a conservative guard, not presented as a
solution of the constrained optimisation problem. See the research protocol.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.stats import rankdata
from .alpha_policy import project, decide


@dataclass(frozen=True)
class ControlSpec:
    family: str = 'C'
    half_life: float = 3.
    risk_aversion: float = 8.
    nominal_penalty: float = .004
    rank_budget: float = .6
    rank_preference: float = .02
    inertia: float = .016
    turnover_24h: float = .5
    budget_shadow: float = .0002
    fee: float = .0002
    reserve_exit_fee: bool = True
    minimum_trade: float = .003
    review_hours: int = 4
    fixed_expiry: bool = False
    no_friction: bool = False

    def __post_init__(self):
        if self.family not in ('A', 'B', 'C', 'D'):
            raise ValueError('unknown control family')
        if self.half_life <= 0 or self.nominal_penalty <= 0 or self.turnover_24h <= 0:
            raise ValueError('positive scales required')
        if min(self.rank_budget, self.rank_preference, self.inertia, self.fee,
               self.minimum_trade, self.risk_aversion, self.budget_shadow) < 0:
            raise ValueError('negative control parameters')
        if self.review_hours < 1:
            raise ValueError('positive review clock required')


def prox_two_costs(v, old, trade_cost, exit_cost, lipschitz):
    """Exact scalar prox of c|w-old|+e|w|, vectorised across assets."""
    low = np.minimum(old, 0.)
    high = np.maximum(old, 0.)
    c, e, L = trade_cost, exit_cost, lipschitz
    # On each of three intervals both absolute-value derivatives are constant.
    left = np.minimum(v + (c + e) / L, low)
    middle = np.clip(v - (e - c) * np.sign(old) / L, low, high)
    right = np.maximum(v - (c + e) / L, high)
    candidates = np.stack((left, middle, right))
    obj = .5 * L * (candidates - v)**2 + c * np.abs(candidates-old) + e * np.abs(candidates)
    return np.take_along_axis(candidates, obj.argmin(0)[None], axis=0)[0]


def solve_control(A, driver, old, inertia, cost, exit_cost, max_iter=100, tol=2e-7):
    """Minimise .5w'Aw-driver'w+.5k||w-old||²+c||w-old||1+e||w||1.

    A is symmetric positive definite. The row-sum bound is a safe Lipschitz
    bound; accelerated updates and warm starts avoid a heavyweight solver.
    Return the infinity norm of the proximal stationarity residual as evidence.
    """
    n = len(old)
    Q = A + np.eye(n)*inertia
    b = driver + inertia*old
    L = float(np.abs(Q).sum(1).max())
    w = old.copy(); y = w.copy(); momentum = 1.
    for iteration in range(max_iter):
        new = prox_two_costs(y-(Q@y-b)/L, old, cost, exit_cost, L)
        next_momentum = (1+np.sqrt(1+4*momentum**2))/2
        y = new+(momentum-1)/next_momentum*(new-w)
        diff = np.max(np.abs(new-w)); w = new; momentum = next_momentum
        if diff < tol:
            residual = np.max(np.abs(w-prox_two_costs(w-(Q@w-b)/L, old, cost, exit_cost, L)))*L
            if residual < tol*L:
                break
    residual = np.max(np.abs(w-prox_two_costs(w-(Q@w-b)/L, old, cost, exit_cost, L)))*L
    return w, iteration+1, float(residual)


def guard(w, cov, policy, neutral=False):
    """Feasible contraction; all risk uses past hourly covariance."""
    out = project(w, policy.cap, 0. if neutral else policy.cap*policy.net_fraction, policy.coin_cap)
    annual = np.sqrt(max(float(out@cov@out), 0.))*np.sqrt(24*365.25)
    if annual > policy.annual_risk:
        out *= policy.annual_risk/annual
    return out


def rank_target(mu, state, budget, fee):
    """Bounded equal nominal prior; raw dispersion is never normalised away."""
    n = len(mu)
    z = (2*rankdata(mu, method='average')-(n+1))/max(n-1, 1)
    spread = float(np.mean(np.abs(mu-mu.mean()))) if np.ptp(mu)>1e-12 else 0.
    support = spread/(spread+2*fee) if spread > 0 else 0.
    shape = np.tanh(1.5*state)
    shape -= shape.mean()
    # Centring can enlarge an individual coordinate; contraction restores bound.
    shape /= max(1., float(np.abs(shape).max()))
    return budget/n*support*shape, z, support


class ControlDecision:
    """One state per account, only one update for each new prediction timestamp.

    The turnover bucket is an EWMA control preference, not a hard trading ban.
    It accounts for accepted changes with the same post-fee fixed point as the
    share ledger; safe changes bypass both the review clock and the bucket.
    """
    def __init__(self, hours, spec=ControlSpec(), expected_funding=None, logger=None, ledger_fee=None):
        self.hours = np.asarray(hours)
        if len(self.hours)==0 or np.any(np.diff(self.hours)!=1):
            raise ValueError('continuous new hourly predictions required')
        self.spec = spec; self.funding = expected_funding; self.logger = logger
        self.ledger_fee = spec.fee if ledger_fee is None else ledger_fee
        self.last_row = -1; self.filtered = None; self.rank_state = None
        self.debt = 0.; self.last_trade = -10000
        self.history = []; self.targets = []; self.filtered_history = []

    def __call__(self, row, mu, cov, beta, current, age, policy):
        if row != self.last_row+1:
            raise ValueError('duplicate or out-of-order prediction; state cannot accumulate it')
        if not np.isfinite(mu).all() or not np.isfinite(cov).all():
            raise ValueError('finite forecast and past risk required')
        self.last_row = row; s = self.spec; n = len(mu)
        alpha = 1-2**(-1/s.half_life)
        z = (2*rankdata(mu, method='average')-(n+1))/max(n-1, 1)
        self.filtered = mu.copy() if self.filtered is None else (1-alpha)*self.filtered+alpha*mu
        self.rank_state = z.copy() if self.rank_state is None else (1-alpha)*self.rank_state+alpha*z
        rank, _, support = rank_target(mu, self.rank_state, s.rank_budget, s.fee)
        P = np.eye(n)-np.ones((n,n))/n
        A = s.risk_aversion*policy.horizon*cov+np.eye(n)*s.nominal_penalty
        funding = np.zeros(n) if self.funding is None else self.funding[row]*policy.horizon
        driver = self.filtered-funding
        if s.family == 'B':
            A += s.rank_preference*np.eye(n)
            driver = s.rank_preference*rank
        elif s.family == 'C':
            A += s.rank_preference*P
            driver += s.rank_preference*rank
        elif s.family == 'D':
            # Tracking preference, explicitly not another return forecast.
            common = (policy.cap-s.rank_budget)/n*np.tanh((self.filtered-funding).mean()/.0008)
            goal = rank+common
            A += s.rank_preference*np.eye(n)
            driver = s.rank_preference*goal
        self.debt *= np.exp(-1/24)
        shadow = s.budget_shadow*max(0., self.debt/s.turnover_24h-1)
        vol = np.sqrt(max(float(np.diag(cov).mean()), 0.))
        k = s.inertia*(1+min(vol/.005, 3.))
        cost = s.fee+shadow; exit_cost = s.fee if s.reserve_exit_fee else 0.
        if s.no_friction: k=cost=exit_cost=0.
        candidate, iterations, residual = solve_control(A, driver, current, k, cost, exit_cost)
        neutral = s.family=='B'
        desired = guard(candidate, cov, policy, neutral)
        safe = guard(current, cov, policy, neutral)
        target = safe.copy(); reason = np.full(n, 'hold', object)
        risk_change = np.abs(safe-current)>1e-8
        reason[risk_change] = 'risk_limit'
        normal = row-self.last_trade >= s.review_hours or s.no_friction
        if normal:
            delta = desired-safe
            active = np.abs(delta) >= s.minimum_trade
            # Accept as a portfolio basket, avoiding neutrality loss from per-leg gates.
            if active.any():
                target = desired.copy(); reason[:] = 'control_'+s.family
                reason[risk_change] = 'risk_limit'
                self.last_trade = row
        if s.fixed_expiry:
            expired = (age>=policy.horizon)&(np.abs(current)>1e-9)
            target[expired] = 0.; reason[expired] = 'expiry'
            target = guard(target, cov, policy, neutral)
        change = np.abs(target-current)>1e-8
        # Prevent unmarked microscopic changes from altering fixed shares.
        target = np.where(change,target,current)
        after = 1.
        for _ in range(8):
            turnover = float(np.abs(np.where(change,target*after,current)-current).sum())
            after = 1-self.ledger_fee*turnover
        traded = turnover
        self.debt += traded
        self.history.append([traded, self.debt, shadow, support, iterations, residual,
                             float(np.abs(candidate-desired).sum()), int(risk_change.any()),
                             float(np.abs(desired-current).sum())])
        self.targets.append(desired.copy()); self.filtered_history.append(self.filtered.copy())
        if self.logger and (row%720==0 or row==len(self.hours)-1):
            self.logger(row, self.history[-1])
        return target, change, reason

    def specification(self):
        return asdict(self.spec)


class CoreParticipationDecision:
    """Exploratory compatibility version: old strong core plus small D sleeve.

    Core timers start at a real upgrade, while the physical account never opens
    two opposite holdings in one symbol. All ordinary increases share a basket
    clock; edge loss/expiry and joint risk reduction can happen immediately.
    Weak sleeve coefficients scale its target, never a second account NAV.
    """
    def __init__(self, hours, weak_fraction=.1, logger=None, weak_enabled=True):
        if not 0 < weak_fraction < 1:raise ValueError('weak fraction must be inside (0,1)')
        self.hours=np.asarray(hours);self.fraction=weak_fraction;self.logger=logger
        self.weak_enabled=weak_enabled
        self.core=None;self.core_age=None;self.last_normal=-10000
        f=weak_fraction
        self.weak_spec=ControlSpec(family='D',risk_aversion=8/f,nominal_penalty=.004/f,
                                  rank_preference=.02/f,inertia=.016/f,rank_budget=.6*f,
                                  minimum_trade=.003*f,turnover_24h=.5*f)
        self.weak=ControlDecision(hours,self.weak_spec)
        self.targets=[];self.role_history=[];self.history=[];self.filtered_history=[]

    def __call__(self,row,mu,cov,beta,current,age,policy):
        n=len(mu)
        if self.core is None:self.core=np.zeros(n,bool);self.core_age=np.zeros(n,int)
        self.core_age+=self.core
        previous_role=self.core.copy()
        core_current=np.where(previous_role,current,0.)
        core_target,core_change,core_reason=decide(mu,np.zeros(n),cov,beta,core_current,self.core_age,policy)
        weak_policy=replace(policy,cap=policy.cap*self.fraction,coin_cap=policy.coin_cap*self.fraction)
        self.weak.last_trade=self.last_normal
        weak_target,_,_=self.weak(row,mu,cov,beta,np.where(previous_role,0.,current),age,weak_policy)
        weak_target=np.where(np.abs(mu)<=2*policy.design_fee*policy.multiplier,weak_target,0.)
        target=current.copy();reason=np.full(n,'hold',object)
        allowed=row-self.last_normal>=4;normal=False
        for j in range(n):
            core_action=core_change[j] and (previous_role[j] or abs(core_target[j])>1e-9)
            if core_action:
                if previous_role[j] and core_reason[j] in ('expiry','edge_lost'):
                    target[j]=0.;self.core[j]=False;reason[j]='core_'+str(core_reason[j])
                elif previous_role[j] and current[j]*core_target[j]<0:
                    # Close now; any new reverse entry must pass the shared clock.
                    target[j]=core_target[j] if allowed else 0.
                    self.core[j]=abs(target[j])>1e-9;self.core_age[j]=0
                    reason[j]='core_reverse';normal|=allowed
                elif allowed or core_reason[j]=='risk_limit':
                    target[j]=core_target[j];self.core[j]=abs(target[j])>1e-9
                    if not previous_role[j] or current[j]*target[j]<=0:self.core_age[j]=0
                    reason[j]='core_'+str(core_reason[j]);normal|=core_reason[j]!='risk_limit'
                continue
            if previous_role[j]:continue
            if allowed and self.weak_enabled:
                target[j]=weak_target[j]
                if abs(target[j]-current[j])>1e-8:reason[j]='control_E_weak';normal=True
        proposed=target.copy();target=guard(target,cov,policy)
        risk_changed=np.abs(target-proposed)>1e-8;reason[risk_changed]='risk_limit'
        change=np.abs(target-current)>1e-8;target=np.where(change,target,current)
        self.core &= np.abs(target)>1e-9
        if normal and change.any():self.last_normal=row
        self.targets.append(target.copy());self.role_history.append(self.core.copy())
        self.filtered_history.append(self.weak.filtered.copy())
        self.history.append([float(np.abs(target-current).sum()),int(self.core.sum()),int(normal),
                             int(risk_changed.any()),float(np.where(self.core,0.,np.abs(target)).sum()),
                             float(np.max(self.core_age))])
        if self.logger and (row%720==0 or row==len(self.hours)-1):self.logger(row,self.history[-1])
        return target,change,reason

    def specification(self):
        return {'family':'E_core_participation','weak_fraction':self.fraction,'shared_normal_review_hours':4,
                'weak_enabled':self.weak_enabled,
                'old_strong_gate_roundtrip':.0008,'core_fixed_lifetime_hours':12,
                'historical_status':'added after A-D historical inspection, explicitly exploratory'}
