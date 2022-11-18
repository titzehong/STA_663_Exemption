import pymc as pm
import numpy as np


class LGCP_model:
    
    def __init__(self, area_per_cell:float, multi:bool=False, time:bool=True):
        self.area_per_cell=area_per_cell
        self.multi = multi
        self.time = time
        
    def init_model(self, X_vars, cell_counts):
        
        if not self.multi:
            with pm.Model() as self.lgcp_model:
                mu = pm.Normal("mu", sigma=3)

                ls1 = pm.Gamma("ls1", alpha=2, beta=2)
                ls2 = pm.Gamma("ls2", alpha=2, beta=2)
                if self.time:
                    ls3 = pm.Gamma("ls3", alpha=2, beta=2)

                # Specify the covariance functions for each Xi
                cov_x1 = pm.gp.cov.Matern52(1, ls=ls1) + 0.001
                cov_x2 = pm.gp.cov.Matern52(1, ls=ls2) + 0.001
                if self.time:
                    cov_x3 = pm.gp.cov.Matern52(1, ls=ls3) + 0.001
                    
                mean_func = pm.gp.mean.Constant(mu)

            with self.lgcp_model:
                
                if self.time:
                    self.gp = pm.gp.LatentKron(mean_func=mean_func, cov_funcs=[cov_x1, cov_x2, cov_x3])
                else:
                    self.gp = pm.gp.LatentKron(mean_func=mean_func, cov_funcs=[cov_x1, cov_x2])
                
                # Create GP Prior at X_vars
                log_intensity = self.gp.prior("log_intensity", Xs=X_vars)
                
                # Get rate at each point
                intensity = pm.math.exp(log_intensity)
                
                

                rates = intensity * self.area_per_cell
                counts = pm.Poisson("counts", mu=rates, observed=cell_counts)

    
    def sample_model(self, draws: int=1000, chains: int=4, tune: int=1000):
        
        with self.lgcp_model:
            self.trace = pm.sample(draws=draws,
                                   tune=tune,
                                   chains=chains)
    
    
    def get_posterior_predictive(self, Xnew):
        
        if not self.trace:
            raise ValueError('No Saved Traced for the model')

        with self.model:
            fnew = self.gp.conditional("fnew", Xnew)
            ppc = pm.sample_posterior_predictive(self.trace, 200, var_names=["fnew"])