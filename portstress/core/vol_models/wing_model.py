import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import plotly.graph_objects as go

# === WingModel Class ===


class WingModel:
    """
    WingModel fits a parametric implied volatility smile model (Wing Model)
    to option market data. It supports arbitrage-free constraints and can
    be used to interpolate/extrapolate implied volatilities.

    Parameters estimated per smile (expiry):
    - atm_vol: ATM implied volatility
    - skew: slope near ATM
    - put_slope / call_slope: slopes in the wings
    - put_curve / call_curve: convexity in the wings
    - slope_change: transition point from ATM slope to wing slope

    The pandas dataframe must contain the following columns:

    i. The implied volatility ('IV') in %: (float64)
    ii. The strike price ('Strike'): (float64)
    iii. The expiry date ('Date'): (pd.Timestamp)
    iv. The time to maturity ('Tau') in years: (float64)
    v. The forward price ('F'): (float64)

    """

    def __init__(self, df: pd.DataFrame, min_fit: int = 5):
        # Check input format
        assert isinstance(df, pd.DataFrame)
        assert all(col in df.columns for col in ['IV', 'Strike', 'Date', 'Tau', 'F'])

        dfv = df.copy()
        # Calculate total variance
        dfv['TV'] = dfv['Tau'] * (dfv['IV'] / 100.0) ** 2
        dfv['LogM'] = np.log(dfv['Strike'] / dfv['F'])

        # Keep expiries with sufficient data
        valid_dates = dfv.groupby('Date')['IV'].count()
        valid_dates = valid_dates[valid_dates >= min_fit].index
        dfv = dfv[dfv['Date'].isin(valid_dates)]

        self.T = dfv['Date'].unique()
        self.dfv_dic = {t: dfv[dfv['Date'] == t] for t in self.T}
        self.param_labels = ['atm_vol', 'skew', 'put_slope', 'call_slope', 'put_curve', 'call_curve', 'slope_change']

    @staticmethod
    def wing_iv(logm, params):
        # Piecewise definition of the Wing model
        a, s, ps, cs, pc, cc, sw = params
        if logm < -sw:
            x = logm + sw
            return a + s * -sw + ps * x + pc * x**2
        elif logm > sw:
            x = logm - sw
            return a + s * sw + cs * x + cc * x**2
        else:
            return a + s * logm

    @staticmethod
    def wing_vectorized(_, x_array, params):
        # Vectorized version for fitting
        return np.array([WingModel.wing_iv(x, params) for x in x_array])

    def fit(self, no_butterfly=True, no_calendar=True, plots=False):
        """
        Fit the Wing model to each expiry's volatility smile.
        Can enforce no-arbitrage constraints.
        """
        self.param_dic = {}
        initial_guess = [0.2, -0.1, -0.2, 0.1, 0.05, 0.03, 0.1]
        bounds = Bounds([0, -1, -1, 0, 0, 0, 0.01], [1, 1, 0, 1, 1, 1, 0.5])

        for ti, t in enumerate(self.T):
            df = self.dfv_dic[t]
            xdata = df['LogM'].values
            ydata = df['TV'].values
            constraints = []

            # Add butterfly arbitrage constraint (convexity)
            if no_butterfly and len(xdata) > 2:
                constraints.append(
                    NonlinearConstraint(fun=lambda p: self._butterfly_constraint(p, xdata),
                                        lb=0, ub=np.inf)
                )

            # Add calendar arbitrage constraint
            if no_calendar and ti > 0:
                prev_t = self.T[ti - 1]
                if prev_t in self.param_dic:
                    prev_params = np.array(list(self.param_dic[prev_t].values()))
                    constraints.append(
                        NonlinearConstraint(fun=lambda p: self._calendar_constraint(p, prev_params, xdata),
                                            lb=0, ub=np.inf)
                    )

            # Minimize MSE between model and market total variance
            res = minimize(
                self._mse,
                initial_guess,
                args=(xdata, ydata),
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 1000}
            )

            self.param_dic[t] = dict(zip(self.param_labels, res.x))

            # Plot smile
            if plots:
                plt.scatter(xdata, ydata, label='Data')
                plt.plot(xdata, self.wing_vectorized(None, xdata, res.x), label='Wing Fit')
                plt.title(f"Wing Smile - {t}")
                plt.xlabel("Log Moneyness")
                plt.ylabel("Total Variance")
                plt.legend()
                plt.grid(True)
                plt.show()

        return pd.DataFrame.from_dict(self.param_dic, orient='index')

    def _mse(self, params, xdata, ydata):
        y_pred = self.wing_vectorized(None, xdata, params)
        return ((y_pred - ydata) ** 2).mean()

    def _butterfly_constraint(self, params, xdata):
        y = self.wing_vectorized(None, xdata, params)
        return np.diff(y, 2)  # finite difference second derivative

    def _calendar_constraint(self, params_new, params_old, xdata):
        y_new = self.wing_vectorized(None, xdata, params_new)
        y_old = self.wing_vectorized(None, xdata, params_old)
        return y_new - y_old  # ensure TV increases

    def get_iv(self, strike: float, forward: float, expiry: pd.Timestamp) -> float:
        if expiry not in self.param_dic:
            raise ValueError("Expiry not fitted")
        logm = np.log(strike / forward)
        total_var = self.wing_iv(logm, list(self.param_dic[expiry].values()))
        tau = self.dfv_dic[expiry]['Tau'].iloc[0]
        return 100 * np.sqrt(total_var / tau)

# === WingPlot Class ===


class WingPlot:
    """
    3D Plotting class for Wing Model implied volatility surface.
    """

    def __init__(self):
        pass

    def allsmiles(self, wm: WingModel):
        dfv_dic, param_dic, T = wm.dfv_dic, wm.param_dic, wm.T
        fig = go.Figure()

        for ti, t in enumerate(T):
            df = dfv_dic[t]
            x = df['LogM'].values
            z = df['IV'].values
            y = df['Date'].values

            x0, x1 = x[0] * 1.1, x[-1] * 1.1
            dx = (x1 - x0) / 200
            xnew = np.arange(x0, x1, dx)
            tau = df['Tau'].iloc[0]
            model_params = list(param_dic[t].values())
            znew = 100 * np.sqrt(wm.wing_vectorized(None, xnew, model_params) / tau)
            ynew = np.array([t] * len(xnew))

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode='markers', name=t.strftime("%Y-%m-%d"),
                legendgroup=f"{str(ti)}", showlegend=False,
                marker=dict(size=2, color='black')
            ))
            fig.add_trace(go.Scatter3d(
                x=xnew, y=ynew, z=znew, mode='lines', name=t.strftime("%Y-%m-%d"),
                legendgroup=f"{str(ti)}", showlegend=True,
            ))

        fig = self._change_camera(fig)
        fig = self._add_onoff(fig)

        fig.update_layout(title="Wing Model Volatility Surface")
        fig.update_scenes(
            xaxis_title_text='Log Moneyness',
            yaxis_title_text='Time',
            zaxis_title_text='Implied Volatility'
        )
        fig.show()
        return fig

    @staticmethod
    def _change_camera(fig: go.Figure) -> go.Figure:
        fig.update_layout(
            width=800, height=700, autosize=False,
            scene=dict(
                camera=dict(up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.25, y=-1, z=0.25)),
                aspectratio=dict(x=1, y=1, z=0.7),
                aspectmode='manual'
            ),
        )
        return fig

    @staticmethod
    def _add_onoff(fig: go.Figure) -> go.Figure:
        fig.update_layout(dict(updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(args=["visible", "legendonly"], label="Deselect All", method="restyle"),
                    dict(args=["visible", True], label="Select All", method="restyle")
                ]),
                pad={"r": 10, "t": 10},
                showactive=False,
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            ),
        ]))
        return fig


# Sample Use
'''
wm = WingModel(df)
wm.fit(no_butterfly=True, no_calendar=True, plots=True)

wp = WingPlot()
wp.allsmiles(wm)
'''
