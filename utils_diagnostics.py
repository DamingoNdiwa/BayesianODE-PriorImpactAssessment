import pandas as pd

parameters = [
    r'$\alpha$',
    r'$\gamma$',
    r'$\delta$',
    r'$\beta$',
    r'$\hat{u}$',
    r'$\hat{v}$',
    r'$\sigma_{u}$',
    r'$\sigma_{v}$']

parameters1 = [
    r'$\eta$',
    r'$\rho$',
    r'$\sigma$',
    r'$\hat{I}$',
    r'$\hat{E}$',
    r'$\lambda$',
    r'$\phi$']


def create_dataframe(data, model="Lotka"):
    if model == 'seir':
        columns = parameters1
    else:
        columns = parameters
    df = pd.DataFrame(data=data, columns=columns)
    return df