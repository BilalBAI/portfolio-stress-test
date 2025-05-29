import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from uuid import uuid4

import plotly.graph_objects as go
from typing import Tuple, Optional


class WingModel:
    """
    WingModel fits a parametric implied volatility smile model (Wing Model) to option market data
    as per the provided documentation. It supports arbitrage-free constraints and interpolation/extrapolation
    of implied volatilities across expiries.

    Parameters estimated per smile (expiry):
    - vr: Volatility reference (constant part of volatility at central skew point)
    - sr: Slope reference (constant part of slope at central skew point)
    - vcr: Volatility change rate
    - scr: Slope change rate
    - pc: Put curvature
    - cc: Call curvature
    - dc: Down cutoff
    - uc: Up cutoff
    - dsm: Down smoothing range
    - usm: Up smoothing range
    - ssr: Skew swimmingness rate (0% to 100%)
    - ref_price: Reference price for volatility and slope adjustments

    The pandas dataframe must contain:
    - IV: Implied volatility in % (float64)
    - Strike: Strike price (float64)
    - Date: Expiry date (pd.Timestamp)
    - Tau: Time to maturity in years (float64)
    - F: Forward price (float64)
    """

    def __init__(self, df: pd.DataFrame, min_fit: int = 3, time_weighted: bool = False):
        # Input validation
        assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
        required_cols = ['IV', 'Strike', 'Date', 'Tau', 'F']
        assert all(col in df.columns for col in required_cols), f"Missing required columns: {required_cols}"

        # Check data types
        assert df['Date'].dtype == 'datetime64[ns]', "'Date' column must be pd.Timestamp"
        assert all(df[col].dtype == 'float64' for col in ['IV', 'Strike',
                   'Tau', 'F']), "Numerical columns must be float64"

        # Check for valid values
        assert (df['IV'] >= 0).all(), "IV must be non-negative"
        assert (df['Tau'] > 0).all(), "Tau must be positive"
        assert (df['F'] > 0).all(), "Forward price must be positive"

        # Handle NaN values
        df = df.dropna(subset=required_cols)

        self.time_weighted = time_weighted
        dfv = df.copy()
        # Calculate total variance and log moneyness
        dfv['TV'] = dfv['Tau'] * (dfv['IV'] / 100.0) ** 2
        dfv['LogM'] = np.log(dfv['Strike'] / dfv['F'])
        if time_weighted:
            dfv['LogM'] = dfv['LogM'] / np.sqrt(np.maximum(dfv['Tau'], 1 / 365))

        # Keep expiries with sufficient data
        valid_dates = dfv.groupby('Date')['IV'].count()
        valid_dates = valid_dates[valid_dates >= min_fit].index
        dfv = dfv[dfv['Date'].isin(valid_dates)]

        self.T = dfv['Date'].unique()
        self.dfv_dic = {t: dfv[dfv['Date'] == t] for t in self.T}
        self.param_labels = ['vr', 'sr', 'vcr', 'scr', 'pc', 'cc', 'dc', 'uc', 'dsm', 'usm', 'ssr', 'ref_price']

    def _calculate_current_params(self, params, atm, ref_price):
        vr, sr, vcr, scr, pc, cc, dc, uc, dsm, usm, ssr, _ = params
        # Calculate current forward price (F)
        if ssr == 1.0:
            F = atm
        elif ssr == 0.0:
            F = ref_price
        else:
            F = (atm ** ssr) * (ref_price ** (1 - ssr))

        # Calculate current volatility and slope
        vc = vr - vcr * ssr * (atm - ref_price) / ref_price
        sc = sr - scr * ssr * (atm - ref_price) / ref_price
        return F, vc, sc

    def wing_iv(self, logm, params, tau=None):
        vr, sr, vcr, scr, pc, cc, dc, uc, dsm, usm, ssr, ref_price = params
        atm = ref_price  # Use ref_price as default ATM for calculation
        F, vc, sc = self._calculate_current_params(params, atm, ref_price)

        # Adjust logm for time-weighted model
        x = logm
        if self.time_weighted and tau is not None:
            x = logm / np.sqrt(np.maximum(tau, 1 / 365))

        # Volatility curve calculation
        if dc * (1 + dsm) < x <= dc:
            return (vc - (1 + 1 / dsm) * pc * dc**2 - (sc * dc) / (2 * dsm) +
                    (1 + 1 / dsm) * (2 * pc * dc + sc) * x -
                    (sc / dsm + sc / (2 * dc * dsm)) * x**2)
        elif x <= dc * (1 + dsm):
            return vc + dc * (2 + dsm) * (sc / 2) + (1 + dsm) * pc * dc**2
        elif dc < x <= 0:
            return vc + sc * x + pc * x**2
        elif 0 < x <= uc:
            return vc + sc * x + cc * x**2
        elif uc < x <= uc * (1 + usm):
            return (vc - (1 + 1 / usm) * cc * uc**2 - (sc * uc) / (2 * usm) +
                    (1 + 1 / usm) * (2 * cc * uc + sc) * x -
                    (sc / usm + sc / (2 * uc * usm)) * x**2)
        else:  # x > uc * (1 + usm)
            return vc + uc * (2 + usm) * (sc / 2) + (1 + usm) * cc * uc**2

    def wing_vectorized(self, _, x_array, params, tau=None):
        result = np.zeros_like(x_array, dtype=float)
        for i, x in enumerate(x_array):
            result[i] = self.wing_iv(x, params, tau)
        return result

    def _mse(self, params, xdata, ydata, tau):
        y_pred = self.wing_vectorized(None, xdata, params, tau)
        return ((y_pred - ydata) ** 2).mean()

    def _butterfly_constraint(self, params, xdata):
        _, _, _, _, pc, cc, dc, uc, dsm, usm, _, _ = params
        second_deriv = np.zeros_like(xdata, dtype=float)
        second_deriv[xdata <= dc * (1 + dsm)] = 0
        second_deriv[(dc * (1 + dsm) < xdata) & (xdata <= dc)] = -2 * (sc / dsm + sc / (2 * dc * dsm))
        second_deriv[(dc < xdata) & (xdata <= 0)] = 2 * pc
        second_deriv[(0 < xdata) & (xdata <= uc)] = 2 * cc
        second_deriv[(uc < xdata) & (xdata <= uc * (1 + usm))] = -2 * (sc / usm + sc / (2 * uc * usm))
        second_deriv[xdata > uc * (1 + usm)] = 0
        return second_deriv

    def _calendar_constraint(self, params_new, params_old, xdata, tau_new, tau_old):
        y_new = self.wing_vectorized(None, xdata, params_new, tau_new)
        y_old = self.wing_vectorized(None, xdata, params_old, tau_old)
        return y_new - y_old

    def _no_negative_variance_constraint(self, params, xdata, tau):
        return self.wing_vectorized(None, xdata, params, tau)

    def fit(self, no_butterfly=True, no_calendar=True, plots=False):
        """
        Fit the Wing model to each expiry's volatility smile.
        Can enforce no-arbitrage constraints.
        """
        self.param_dic = {}
        # Bounds as per document
        bounds = Bounds(
            lb=[0.05, -np.inf, -np.inf, -np.inf, 0, 0, -np.inf, 0, 0, 0, 0, 0],
            ub=[4.0, np.inf, np.inf, np.inf, np.inf, np.inf, 0, np.inf, np.inf, np.inf, 1.0, np.inf]
        )

        for ti, t in enumerate(sorted(self.T)):
            df = self.dfv_dic[t]
            xdata = df['LogM'].values
            ydata = df['TV'].values
            tau = df['Tau'].iloc[0]
            atm = df['F'].iloc[0]
            # Initial guess
            initial_guess = [df['IV'].mean() / 100, -0.1, 0.2, 0.2, 0.05, 0.03, -0.5, 0.5, 0.5, 0.5, 0.5, atm]
            constraints = []

            # No-negative-variance constraint
            constraints.append(
                NonlinearConstraint(
                    fun=lambda p: self._no_negative_variance_constraint(p, xdata, tau),
                    lb=0, ub=np.inf
                )
            )

            # Butterfly arbitrage constraint
            if no_butterfly:
                constraints.append(
                    NonlinearConstraint(
                        fun=lambda p: self._butterfly_constraint(p, xdata),
                        lb=0, ub=np.inf
                    )
                )

            # Calendar arbitrage constraint
            if no_calendar and ti > 0:
                prev_t = self.T[ti - 1]
                if prev_t in self.param_dic:
                    prev_params = np.array(list(self.param_dic[prev_t].values()))
                    prev_tau = self.dfv_dic[prev_t]['Tau'].iloc[0]
                    constraints.append(
                        NonlinearConstraint(
                            fun=lambda p: self._calendar_constraint(p, prev_params, xdata, tau, prev_tau),
                            lb=0, ub=np.inf
                        )
                    )

            # Minimize MSE
            res = minimize(
                self._mse,
                initial_guess,
                args=(xdata, ydata, tau),
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 1000}
            )

            if not res.success:
                print(f"Warning: Optimization failed for expiry {t}: {res.message}")
                continue

            self.param_dic[t] = dict(zip(self.param_labels, res.x))

            if plots:
                self.plot_smile(t)

        return pd.DataFrame.from_dict(self.param_dic, orient='index')

    def interpolate_params(self, target_date: pd.Timestamp):
        """
        Interpolate skew settings for a target expiry date between known expiries.
        """
        if target_date in self.param_dic:
            return self.param_dic[target_date]

        sorted_dates = sorted(self.T)
        if target_date < sorted_dates[0]:
            return self.param_dic[sorted_dates[0]]
        if target_date > sorted_dates[-1]:
            return self.param_dic[sorted_dates[-1]]

        # Find closest dates
        t1 = max([d for d in sorted_dates if d < target_date])
        t2 = min([d for d in sorted_dates if d > target_date])
        days_t1 = (t1 - pd.Timestamp.today()).days
        days_t2 = (t2 - pd.Timestamp.today()).days
        days_tx = (target_date - pd.Timestamp.today()).days

        # Linear interpolation
        u = (days_t2 - days_tx) / (days_t2 - days_t1)
        params_t1 = np.array(list(self.param_dic[t1].values()))
        params_t2 = np.array(list(self.param_dic[t2].values()))
        interpolated_params = params_t1 * u + params_t2 * (1 - u)
        return dict(zip(self.param_labels, interpolated_params))

    def get_iv(self, strike: float, forward: float, expiry: pd.Timestamp) -> float:
        """
        Calculate implied volatility for a given strike, forward, and expiry.
        """
        if forward <= 0 or strike <= 0:
            raise ValueError("Strike and forward must be positive")

        params = self.interpolate_params(expiry)
        tau = (expiry - pd.Timestamp.today()).days / 365.0
        if tau <= 0:
            raise ValueError("Tau must be positive")

        logm = np.log(strike / forward)
        total_var = self.wing_iv(logm, list(params.values()), tau)
        if total_var < 0:
            raise ValueError("Negative total variance encountered")

        return 100 * np.sqrt(total_var / tau)

    def plot_smile(self, expiry: pd.Timestamp, save_path: str = None):
        """
        Plot the volatility smile for a given expiry.
        """
        if expiry not in self.dfv_dic:
            raise ValueError("Expiry not fitted")
        df = self.dfv_dic[expiry]
        xdata = df['LogM'].values
        ydata = df['TV'].values
        tau = df['Tau'].iloc[0]
        plt.scatter(xdata, ydata, label='Data')
        plt.plot(xdata, self.wing_vectorized(None, xdata, list(self.param_dic[expiry].values()), tau), label='Wing Fit')
        plt.title(f"Wing Smile - {expiry}")
        plt.xlabel("Log Moneyness")
        plt.ylabel("Total Variance")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class WingPlot:
    """
    3D Plotting class for Wing Model implied volatility surface.
    Compatible with the updated WingModel class supporting full parameter set
    and time-weighted option.
    """

    def __init__(self):
        pass

    def allsmiles(self, wm: 'WingModel', xnew_range: Tuple[float, float] = (-0.5, 0.5),
                  points: int = 200, date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None):
        """
        Plot 3D implied volatility surface for all expiries or a specified date range.

        Args:
            wm (WingModel): Fitted WingModel instance.
            xnew_range (tuple): Range for log-moneyness interpolation (min, max).
            points (int): Number of interpolation points per curve.
            date_range (tuple, optional): (start_date, end_date) for interpolated expiries.

        Returns:
            go.Figure: Plotly figure object.
        """
        dfv_dic, param_dic, T = wm.dfv_dic, wm.param_dic, wm.T

        # Check if data is available
        if not T.size or not param_dic:
            raise ValueError("No valid expiries or fitted parameters available")

        fig = go.Figure()
        dates_to_plot = T

        # If date_range is provided, generate interpolated dates
        if date_range is not None:
            start_date, end_date = date_range
            if start_date >= end_date:
                raise ValueError("start_date must be before end_date")
            days = pd.date_range(start=start_date, end=end_date, freq='D')
            dates_to_plot = [d for d in days if d not in T or d in param_dic]

        for t in sorted(dates_to_plot):
            # Get data for the expiry
            if t in dfv_dic:
                df = dfv_dic[t]
                x = df['LogM'].values
                z = df['IV'].values
                y = df['Date'].values
                tau = df['Tau'].iloc[0]
                atm = df['F'].iloc[0]
            else:
                # Interpolated expiry
                tau = (t - pd.Timestamp.today()).days / 365.0
                if tau <= 0:
                    print(f"Warning: Skipping expiry {t} due to non-positive Tau ({tau})")
                    continue
                x = z = y = np.array([])  # No market data for interpolated dates
                atm = param_dic.get(T[0], {}).get('ref_price', 100.0)  # Fallback ATM

            # Get or interpolate parameters
            model_params = param_dic.get(t, wm.interpolate_params(t))
            if not model_params:
                print(f"Warning: No parameters for expiry {t}, skipping")
                continue
            model_params = list(model_params.values())

            # Generate new points for smooth curve
            x0, x1 = xnew_range
            dx = (x1 - x0) / (points - 1)
            xnew = np.arange(x0, x1 + dx, dx)

            if tau <= 0:
                print(f"Warning: Skipping expiry {t} due to non-positive Tau ({tau})")
                continue

            # Calculate total variance
            total_var = wm.wing_vectorized(None, xnew, model_params, tau)

            # Check for negative total variance
            if (total_var < 0).any():
                print(f"Warning: Negative total variance for expiry {t}, filtering invalid points")
                valid_mask = total_var >= 0
                xnew = xnew[valid_mask]
                total_var = total_var[valid_mask]
                if xnew.size == 0:
                    print(f"Warning: No valid points for expiry {t}, skipping")
                    continue

            try:
                znew = 100 * np.sqrt(total_var / tau)
            except (RuntimeWarning, ValueError) as e:
                print(f"Warning: Failed to compute IV for expiry {t}: {e}")
                continue

            ynew = np.array([t] * len(xnew))

            # Add market data points (if available)
            if len(x) > 0:
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z, mode='markers', name=t.strftime("%Y-%m-%d"),
                    legendgroup=f"{t}", showlegend=False,
                    marker=dict(size=2, color='black'),
                    visible=True
                ))

            # Add fitted or interpolated smile curve
            fig.add_trace(go.Scatter3d(
                x=xnew, y=ynew, z=znew, mode='lines', name=t.strftime("%Y-%m-%d"),
                legendgroup=f"{t}", showlegend=True,
                visible=True
            ))

        if len(fig.data) == 0:
            print("Warning: No valid curves to plot")
            return fig

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
        """Configure 3D plot camera settings."""
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
        """Add buttons to toggle traces on/off."""
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


# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        'IV': [20.0, 22.0, 25.0, 18.0, 21.0, 23.0],
        'Strike': [100.0, 110.0, 120.0, 100.0, 110.0, 120.0],
        'Date': [pd.Timestamp('2025-12-31')] * 3 + [pd.Timestamp('2026-03-31')] * 3,
        'Tau': [0.5] * 3 + [0.75] * 3,
        'F': [105.0] * 6
    })

    # Initialize and fit model
    wm = WingModel(data, min_fit=3)
    wm.fit(no_butterfly=True, no_calendar=False)

    # Plot volatility surface
    wp = WingPlot()
    wp.allsmiles(wm, xnew_range=(-0.5, 0.5), points=200)
