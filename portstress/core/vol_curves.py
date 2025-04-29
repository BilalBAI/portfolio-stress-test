import time
import datetime
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize, Bounds
from typing import Union
 
import plotly.graph_objects as go
 
class SVIModel(object):
    """
    Reference: https://sellersgaard.github.io/blog/2023/svi/
   
    This class fits Gatheral's Stochastic Volatility Inspired (SVI) model to a pandas dataframe of
    implied volatility data. The pandas dataframe must contain the following columns:
   
    i. The implied volatility ('IV') in %: (float64)
    ii. The strike price ('Strike'): (float64)
    iii. The expiry date ('Date'): (pd.Timestamp)
    iv. The time to maturity ('Tau') in years: (float64)
    v. The forward price ('F'): (float64)
   
    The vol surface is fit smile-by-smile using a Sequential Least SQuares Programming optimizer which has
    the option of preventing static arbitrage (butterfly and calendar arbitrage).
    To perform the calibration, call the fit method.
    The calibrated parameters are saved in the dictionary 'param_dic', but can also be returned as a pandas dataframe.
   
    Simon Ellersgaard Nielsen, 2023-11-12
    """
   
    def __init__(self, df: pd.DataFrame, min_fit: int = 5):
        """
        df: Volatility dataframe. Must contain ['IV','Strike','Date','Tau','F']
        min_fit: The minimum number of observations per smile required.
        """
       
        assert type(df) == pd.DataFrame
        assert 'IV' in df.columns
        assert 'Strike' in df.columns
        assert 'Date' in df.columns
        assert 'Tau' in df.columns
        assert 'F' in df.columns
       
        dfv = df.copy()
       
        dfv['TV'] = dfv['Tau']*((dfv['IV']/100.0)**2)
        dfvs = dfv.groupby('Date')['IV'].count()
        dfvs = dfvs[dfvs>=min_fit]
        dfv = dfv[dfv['Date'].isin(dfvs.index)]
       
        dfv['LogM'] = np.log(dfv['Strike']/dfv['F'])
        dfv = dfv.sort_values(['Date','Strike'])
       
        self.T = dfv['Date'].unique()
        self.tau = dfv['Tau'].unique()
        self.F = dfv['F'].unique()
       
        self.dfv_dic = {t: dfv[dfv['Date']==t] for t in self.T}
        self.lbl = ['a', 'b', 'rho', 'm', 'sigma']
       
    def fit(self, no_butterfly: bool=True, no_calendar: bool=True, plotsvi: bool=False, **kwargs) -> pd.DataFrame:
        """
        Fits SVI model smile-by-smile to the data. If no no-arbitrage constraints are enforced the curves are fit using
        SciPys curve_fit. If no-arbitrage is required we use SciPy's sequential least squares minimization.
        """
       
        ϵ = 1e-6
        bnd = kwargs.pop('bnd', ([-np.inf,0,-1+ϵ,-np.inf,0], [np.inf,1,1-ϵ,np.inf,np.inf]))
        p0 = kwargs.pop('p0', [0.1,  0.1, 0, 0,  0.1])      
        self.no_butterfly = no_butterfly
        self.no_calendar = no_calendar
       
        # Loop over the individual smiles
        self.param_dic = {}
        for ti, t in enumerate(self.T):
           
            self.ti = ti
           
            self.xdata = self.dfv_dic[t]['LogM']
            self.ydata = self.dfv_dic[t]['TV']
           
            if (not no_butterfly) & (not no_calendar):
                maxfev = kwargs.pop('maxfev',1e6)
                popt, _ = curve_fit(self.svi, self.xdata, self.ydata, jac=self.svi_jac, p0=p0,  bounds=bnd, maxfev=maxfev)
            else:
                ineq_cons = self._no_arbitrage()
                tol = kwargs.pop('tol', 1e-50)
                res = minimize(self.svi_mse,
                               p0,
                               args=(self.xdata, self.ydata),
                               method='SLSQP',
                               jac=self.svi_mse_jac,
                               constraints=ineq_cons,  
                               bounds=Bounds(bnd[0],bnd[1]),
                               tol=tol,
                               options=kwargs)
                popt = res.x
           
            self.param_dic[t] = dict(zip(self.lbl, popt))
   
            if plotsvi:
                fig, ax = plt.subplots(1,1)
                ax.scatter(self.xdata,self.ydata)
                yest = self.svi(self.xdata, *popt)
                ax.plot(self.xdata,yest)
                ax.set_title(t)
       
        return pd.DataFrame.from_dict(self.param_dic, orient='index')
           
    def _no_arbitrage(self):
        """
        No arbitrage constraints on the SVI fit
        """
       
        ineq_cons = []
 
        if self.no_butterfly:  
            ineq_cons.append({'type': 'ineq',
                 'fun' : self.svi_butterfly,
                 'jac' : self.svi_butterfly_jac})
           
        if self.no_calendar:
            if self.ti > 0:
                xv = self.xdata.values
                xv = np.append(np.append(np.array(xv[0]-2), xv),np.array(xv[-1]+2))
                pv = np.array([self.param_dic[self.T[self.ti-1]][i] for i in self.lbl])
                ineq_cons.append({
                    'type': 'ineq',
                    'fun': self.svi_calendar,
                    'jac': self.svi_calendar_jac,
                    'args': (pv, xv),
                })
           
        return ineq_cons
       
    @staticmethod
    def svi_vol_calc(strike, a, b, rho, m, sigma, F, tt):
        d = rho * b * sigma
        c = b * sigma
        y = (np.log(strike / F) - m) / sigma
        svi_vol = np.sqrt((a + d * y + c * np.sqrt(y**2 + 1)) / tt)
        return svi_vol
 
    @staticmethod
    def svi(k: Union[np.array, float], a: float, b: float, rho: float, m: float, sigma: float) -> Union[np.array, float]:
        """
        SVI parameterisation of the total variance curve
        """
        return a + b*( rho*(k-m) + np.sqrt( (k-m)**2 + sigma**2 ))
   
    @staticmethod
    def dsvi(k: Union[np.array, float], a: float, b: float, rho: float, m: float, sigma: float) -> Union[np.array, float]:
        """
        d(SVI)/dk
        """
        return b*rho + (b*(k-m))/np.sqrt( (k-m)**2 + sigma**2 )
       
    @staticmethod
    def d2svi(k: Union[np.array, float], a: float, b: float, rho: float, m: float, sigma: float) -> Union[np.array, float]:
        """
        d^2(SVI)/dk^2
        """
        return b*sigma**2/( (k-m)**2 + sigma**2 )**(1.5)
   
    def q_density(self, k: Union[np.array, float], a: float, b: float, rho: float, m: float, sigma: float) -> Union[np.array, float]:
        """
        Gatheral's risk neutral density function
        """
        params = np.array([a, b, rho, m, sigma])
        w = self.svi(k, *params)
        dw = self.dsvi(k, *params)
        d2w = self.d2svi(k, *params)
        d = -k/np.sqrt(w) - np.sqrt(w)/2
        g = (1-k*dw/(2*w))**2 - 0.25*((dw)**2)*(1/w + 0.25) + 0.5*d2w
        return g/np.sqrt(2*np.pi*w)*np.exp(-0.5*d**2)
   
    @staticmethod
    def svi_jac(k: Union[np.array, float], a: float, b: float, rho: float, m: float, sigma: float) -> np.array:
        """
        Jacobian of the SVI parameterisation
        """
        dsda = np.ones(len(k))
        dsdb = rho*(k-m)+np.sqrt((k-m)**2+sigma**2)
        dsdrho = b*(k-m)
        dsdm = b*(-rho+(m-k)/np.sqrt(sigma**2 + (k-m)**2))
        dsdsigma = b*sigma/np.sqrt(sigma**2 + (k-m)**2)
        return np.array([dsda,dsdb,dsdrho,dsdm,dsdsigma]).T
 
    def svi_mse(self, params: np.array, xdata: np.array, ydata: np.array) -> np.array:
        """
        Sum of squared errors of the SVI model
        """
        y_pred = self.svi(xdata, *params)
        return ((y_pred - ydata)**2).sum()
   
    def svi_mse_jac(self, params: np.array, xdata: np.array, ydata: np.array) -> np.array:
        """
        Jacobian of the sum of squared errors
        """
        y_pred = self.svi(xdata, *params)
        jac = self.svi_jac(xdata, *params)
        return ((y_pred - ydata).T.values*(jac).T).sum(axis=1)
 
    @staticmethod
    def svi_butterfly(params: np.array) -> np.array:
        """
        SVI butterfly arbitrage constraints (all must be >= 0)
        """
        a, b, rho, m, sigma = params
        c1 = (a-m*b*(rho+1))*(4-a+m*b*(rho+1))-(b**2)*(rho+1)**2
        c2 = (a-m*b*(rho-1))*(4-a+m*b*(rho-1))-(b**2)*(rho-1)**2
        c3 = 4-(b**2)*(rho+1)**2
        c4 = 4-(b**2)*(rho-1)**2
        return np.array([c1,c2,c3,c4])
 
    @staticmethod
    def svi_butterfly_jac(params: np.array) -> np.array:
        """
        Jacobian of SVI butterfly constraints
        """
        a, b, rho, m, sigma = params
        dc1da = -2*a+2*b*m*(rho+1)+4
        dc1db = -2*b*(rho+1)**2+m*(a-b*m*(rho+1))*(rho+1)-m*(rho+1)*(-a+b*m*(rho+1)+4)
        dc1drho = -(b**2)*(2*rho+2)+b*m*(a-b*m*(rho+1))-b*m*(-a+b*m*(rho+1)+4)
        dc1dm = b*(a-b*m*(rho+1))*(rho+1)-b*(rho+1)*(-a+b*m*(rho+1)+4)
        dc2da = -2*a+2*b*m*(rho-1)+4
        dc2db = -2*b*(rho-1)**2+m*(a-b*m*(rho-1))*(rho-1)-m*(rho-1)*(-a+b*m*(rho-1)+4)
        dc2drho = -(b**2)*(2*rho-2)+b*m*(a-b*m*(rho-1))-b*m*(-a+b*m*(rho-1)+4)
        dc2dm = b*(a-b*m*(rho-1))*(rho-1)-b*(rho-1)*(-a+b*m*(rho-1)+4)
        dc3db = -2*b*(rho+1)**2
        dc3drho = -(b**2)*(2*rho+2)
        dc4db = -2*b*(rho-1)**2
        dc4drho = -(b**2)*(2*rho-2)
        dc1dsigma = dc2dsigma = dc3da = dc3dm = dc3dsigma = dc4da = dc4dm = dc4dsigma = 0
        return np.array([[dc1da, dc1db, dc1drho, dc1dm, dc1dsigma],
                         [dc2da, dc2db, dc2drho, dc2dm, dc2dsigma],
                         [dc3da, dc3db, dc3drho, dc3dm, dc3dsigma],
                         [dc4da, dc4db, dc4drho, dc4dm, dc4dsigma]])
 
    def svi_calendar(self, params: np.array, params_old: np.array, k: float) -> float:
        """
        SVI calendar arbitrage constraint (must be >= 0)
        """
        return self.svi(k, *params) - self.svi(k, *params_old)
 
    def svi_calendar_jac(self, params: np.array, params_old: np.array, k: float) -> np.array:
        """
        Jacobian of SVI calendar constraint
        """
        return self.svi_jac(k, *params)
 
 
class SVIPlot(SVIModel):
 
    def __init__(self):
        pass
 
    def allsmiles(self, sv: type(SVIModel)):
        """
        Plots all volatility smiles in a single 3d figure (data and fit)
        """
 
        dfv_dic, param_dic, T, tau = sv.dfv_dic, sv.param_dic, sv.T, sv.tau
 
        fig = go.Figure()
        for ti, t in enumerate(T):
 
            x = dfv_dic[t]['LogM'].values
            z = dfv_dic[t]['IV'].values
            y = dfv_dic[t]['Date'].values
 
            print(param_dic[t])
 
            x0, x1 =  x[0]*1.1, x[-1]*1.1
            dx = (x1-x0)/200
            xnew = np.arange(x0,x1,dx)
            znew = 100*np.sqrt(self.svi(xnew, **param_dic[t])/tau[ti])
            #znew = 100*np.sqrt(self.svi(xnew, param_dic[t]['a'], param_dic[t]['b'], param_dic[t]['ρ'], param_dic[t]['m'], param_dic[t]['σ'])/tau[ti])
            ynew = np.array([t]*len(xnew))
            #ynew[-1] += timedelta(seconds=1)
 
            # See https://plotly.com/python/legend/#grouped-legend-items
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode='markers', name = t.strftime("%Y-%m-%d"),
                legendgroup=f"{str(ti)}", showlegend=False,
                marker=dict(
                    size=2,
                    color='Black',
                    #colorscale='PuRd',
                )))
            fig.add_trace(go.Scatter3d(
                x=xnew, y=ynew, z=znew, mode='lines', name = t.strftime("%Y-%m-%d"), legendgroup=f"{str(ti)}", showlegend=True,
            ))
 
        fig = self._change_camera(fig)
 
        fig = self._add_onoff(fig)
 
        fig.update_layout(title="Volatility Surface")
        fig.update_scenes(xaxis_title_text='Log Moneyness',
                          yaxis_title_text='Time',
                          zaxis_title_text='Implied Volatility')
 
        fig.show()
        return fig
 
    @staticmethod
    def _change_camera(fig: go.Figure()) -> go.Figure():
        """
        Adjusts initial view of vol surface
        """
 
        fig.update_layout(
            width=800,
            height=700,
            autosize=False,
            scene=dict(
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.25, y=-1, z=0.25)
                ),
                aspectratio = dict( x=1, y=1, z=0.7 ),
                aspectmode = 'manual'
            ),
        )
        return fig
 
    @staticmethod
    def _add_onoff(fig: go.Figure()) -> go.Figure():
        """
        Add select/deselect all buttons to plot
        """
 
        fig.update_layout(dict(updatemenus=[
            dict(
                type = "buttons",
                direction = "left",
                buttons=list([
                    dict(
                        args=["visible", "legendonly"],
                        label="Deselect All",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", True],
                        label="Select All",
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=False,
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            ),
        ]
        ))
        return fig