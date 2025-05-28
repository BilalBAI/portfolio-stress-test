import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import plotly.graph_objects as go


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import minimize, Bounds, NonlinearConstraint


class WingModel:
    """
    WingModel fits a parametric implied volatility smile model (Wing Model)
    to option market data. It supports arbitrage-free constraints and can
    be used to interpolate/extrapolate implied volatilities.

    Parameters estimated per smile (expiry):
    - atm_vol: ATM implied volatility
    - skew: slope near ATM and in the wings (ensures continuity)
    - put_curve / call_curve: convexity in the wings
    - slope_change: transition point from ATM to wing regions

    The pandas dataframe must contain the following columns:
    - IV: Implied volatility in % (float64)
    - Strike: Strike price (float64)
    - Date: Expiry date (pd.Timestamp)
    - Tau: Time to maturity in years (float64)
    - F: Forward price (float64)
    """

    def __init__(self, df: pd.DataFrame, min_fit: int = 3):
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

        dfv = df.copy()
        # Calculate total variance and log moneyness
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
        # Piecewise definition of the Wing model with continuous slopes
        a, s, _, _, pc, cc, sw = params  # put_slope and call_slope unused (set to skew)
        if logm < -sw:
            x = logm + sw
            return a + s * -sw + s * x + pc * x**2
        elif logm > sw:
            x = logm - sw
            return a + s * sw + s * x + cc * x**2
        else:
            return a + s * logm

    @staticmethod
    def wing_vectorized(_, x_array, params):
        # Vectorized Wing model
        a, s, _, _, pc, cc, sw = params
        result = np.zeros_like(x_array, dtype=float)
        mask_put = x_array < -sw
        mask_atm = (-sw <= x_array) & (x_array <= sw)
        mask_call = x_array > sw

        result[mask_put] = a + s * (-sw) + s * (x_array[mask_put] + sw) + pc * (x_array[mask_put] + sw)**2
        result[mask_atm] = a + s * x_array[mask_atm]
        result[mask_call] = a + s * sw + s * (x_array[mask_call] - sw) + cc * (x_array[mask_call] - sw)**2
        return result

    def _mse(self, params, xdata, ydata):
        # Mean squared error for optimization
        y_pred = self.wing_vectorized(None, xdata, params)
        return ((y_pred - ydata) ** 2).mean()

    def _butterfly_constraint(self, params, xdata):
        # Analytical second derivative for convexity (butterfly arbitrage)
        _, _, _, _, pc, cc, sw = params
        second_deriv = np.zeros_like(xdata, dtype=float)
        second_deriv[xdata < -sw] = 2 * pc
        second_deriv[(xdata >= -sw) & (xdata <= sw)] = 0
        second_deriv[xdata > sw] = 2 * cc
        return second_deriv  # Must be non-negative

    def _calendar_constraint(self, params_new, params_old, xdata):
        # Ensure total variance increases with time to maturity
        y_new = self.wing_vectorized(None, xdata, params_new)
        y_old = self.wing_vectorized(None, xdata, params_old)
        return y_new - y_old  # Must be non-negative

    def _no_negative_variance_constraint(self, params, xdata):
        # Ensure total variance is non-negative
        return self.wing_vectorized(None, xdata, params)

    def fit(self, no_butterfly=True, no_calendar=True, plots=False):
        """
        Fit the Wing model to each expiry's volatility smile.
        Can enforce no-arbitrage constraints.
        """
        self.param_dic = {}
        bounds = Bounds([0, -1, -1, -1, 0, 0, 0.01], [2, 1, 1, 1, 1, 1, 0.5])

        for ti, t in enumerate(self.T):
            df = self.dfv_dic[t]
            xdata = df['LogM'].values
            ydata = df['TV'].values
            # Dynamic initial guess based on data
            initial_guess = [df['IV'].mean() / 100, -0.1, 0, 0, 0.05, 0.03, 0.1]
            constraints = []

            # No-negative-variance constraint
            constraints.append(
                NonlinearConstraint(
                    fun=lambda p: self._no_negative_variance_constraint(p, xdata),
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
                    constraints.append(
                        NonlinearConstraint(
                            fun=lambda p: self._calendar_constraint(p, prev_params, xdata),
                            lb=0, ub=np.inf
                        )
                    )

            # Minimize MSE
            res = minimize(
                self._mse,
                initial_guess,
                args=(xdata, ydata),
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

    def get_iv(self, strike: float, forward: float, expiry: pd.Timestamp) -> float:
        # Calculate implied volatility for a given strike, forward, and expiry
        if expiry not in self.param_dic:
            raise ValueError("Expiry not fitted")
        if forward <= 0 or strike <= 0:
            raise ValueError("Strike and forward must be positive")
        logm = np.log(strike / forward)
        total_var = self.wing_iv(logm, list(self.param_dic[expiry].values()))
        tau = self.dfv_dic[expiry]['Tau'].iloc[0]
        if total_var < 0:
            raise ValueError("Negative total variance encountered")
        if tau <= 0:
            raise ValueError("Tau must be positive")
        return 100 * np.sqrt(total_var / tau)

    def plot_smile(self, expiry: pd.Timestamp, save_path: str = None):
        # Plot the volatility smile for a given expiry
        if expiry not in self.param_dic:
            raise ValueError("Expiry not fitted")
        df = self.dfv_dic[expiry]
        xdata = df['LogM'].values
        ydata = df['TV'].values
        plt.scatter(xdata, ydata, label='Data')
        plt.plot(xdata, self.wing_vectorized(None, xdata, list(self.param_dic[expiry].values())), label='Wing Fit')
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
    """

    def __init__(self):
        pass

    def allsmiles(self, wm: WingModel, xnew_range: tuple = (-0.5, 0.5), points: int = 200):
        """
        Plot 3D implied volatility surface for all expiries.

        Args:
            wm (WingModel): Fitted WingModel instance.
            xnew_range (tuple): Range for log-moneyness interpolation (min, max).
            points (int): Number of interpolation points per curve.

        Returns:
            go.Figure: Plotly figure object.
        """
        dfv_dic, param_dic, T = wm.dfv_dic, wm.param_dic, wm.T

        # Check if data is available
        if not T.size or not param_dic:
            raise ValueError("No valid expiries or fitted parameters available")

        fig = go.Figure()

        for ti, t in enumerate(T):
            df = dfv_dic.get(t)
            if df is None or df.empty:
                print(f"Warning: No data for expiry {t}")
                continue

            x = df['LogM'].values
            z = df['IV'].values
            y = df['Date'].values

            # Skip if insufficient data
            if len(x) < 2:
                print(f"Warning: Insufficient data points for expiry {t}, skipping")
                continue

            # Generate new points for smooth curve
            x0, x1 = xnew_range  # Use fixed range for consistency
            dx = (x1 - x0) / (points - 1)
            xnew = np.arange(x0, x1 + dx, dx)  # Ensure endpoint inclusion
            tau = df['Tau'].iloc[0]

            if tau <= 0:
                print(f"Warning: Skipping expiry {t} due to non-positive Tau ({tau})")
                continue

            model_params = param_dic.get(t)
            if model_params is None:
                print(f"Warning: No fitted parameters for expiry {t}, skipping")
                continue
            model_params = list(model_params.values())

            total_var = wm.wing_vectorized(None, xnew, model_params)

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

            # Add market data points
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode='markers', name=t.strftime("%Y-%m-%d"),
                legendgroup=f"{str(ti)}", showlegend=False,
                marker=dict(size=2, color='black'),
                visible=True
            ))
            # Add fitted smile curve
            fig.add_trace(go.Scatter3d(
                x=xnew, y=ynew, z=znew, mode='lines', name=t.strftime("%Y-%m-%d"),
                legendgroup=f"{str(ti)}", showlegend=True,
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
