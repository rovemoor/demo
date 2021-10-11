"""Script to plot histogram of returns wrt metric greater than value over timeframe after the metric is observed

Conform PEP8, except for lines are extended to 120 characters

The function would have 3 inputs, metric, value, and timeframe. Metric would allow me to choose a ratio, value would be
the minimum value to be observed to be used to classify the stock is in the universe I am interested in, and timeframe
is the length of time over which to calculate the price change into the future.For example, I want to know what the
distribution of returns has been (histogram of price changes) for stocks that have had price/epsntm [Metric, string]
ratios greater than 30 [value, float] over a 6 month [timeframe, integer] timespan after the metric is observed.
This is one histogram, not one per stock.Use the following metrics: price/epsNTM, entrVal/ebitdaNTM, entrVal/salesNTM,
roe and pB. The output would be a chart and a data.frame of binned returns and their frequencies.

"""
import pandas as pd
import numpy as np


def read_factor_data():
    """Read the factor data"""
    return pd.read_csv(r'D:\python\factor\interviewAssignment.csv', parse_dates=['date'])


def pipeline_factor(factor):
    """Preprocess the factor with sort, fill na, and adding additional metrics"""
    factor = factor.sort_values(['ticker_exchange', 'date'])
    # TODO: Forward fill all the metrics. This step can be further refined or removed
    factor = factor[['ticker_exchange']].join(factor.groupby(['ticker_exchange']).ffill())
    numerator = ['price', 'entrval', 'entrval']
    denominator = ['epsntm', 'ebitdantm', 'salesntm']
    for num, denom in zip(numerator, denominator):
        factor[f'{num}/{denom}'] = factor[num] / factor[denom]
    return factor


def _check_factor_input(metric, value, timeframe):
    if (type(metric) != str) or \
            (metric.lower() in ('price/epsNTM', 'entrVal/ebitdaNTM', 'entrVal/salesNTM', 'roe', 'pB')):
        raise ValueError('Metric should be in price/epsNTM, entrVal/ebitdaNTM, entrVal/salesNTM, roe and pB')
    try:
        value = float(value)
    except TypeError:
        raise TypeError('value input should be float.')
    if type(timeframe) != int:
        raise TypeError('timeframe input should be int.')
    return value


def return_hist_factor(metric, value, timeframe):
    """Plot histogram of returns wrt metric greater than value over timeframe after the metric is observed

    Parameters
    ----------
    metric: {'price/epsNTM', 'entrVal/ebitdaNTM', 'entrVal/salesNTM', 'roe', 'pB'}
    value: float, optional
        The threshold to filter metric greater than value
    timeframe: int
        The timespan after the metric is observed

    Returns
    -------
    DataFrame
        returns bins and their frequencies and also plot the histogram

    """
    value = _check_factor_input(metric, value, timeframe)
    factor = read_factor_data()
    factor.columns = factor.columns.str.lower()
    factor = pipeline_factor(factor)
    start_mth = sorted(factor['date'].unique())[-timeframe-1]
    # Note, the date is not all month end. The timeframe slice on date can be further refined
    # Filter the stocks with metric>value with timeframe relative to latest month
    stocks = factor.loc[(factor[metric.lower()] > value)&(factor['date'] == start_mth), 'ticker_exchange'].unique()
    # Retrieved the point in time future returns of the filtered stocks
    returns = factor.loc[(factor['ticker_exchange'].isin(stocks))&(factor['date'] > start_mth), 'pricereturn1m'].dropna()
    count, division = np.histogram(returns)
    returns.plot.hist(title=f'{timeframe} months future return histogram wrt {metric}', bins=division)
    return pd.DataFrame(count, index=zip(division[:-1], division[1:]), columns=['frequencies'])


if __name__ == '__main__':
    # Here we plot one example
    metric, value, timeframe = 'price/epsNTM', 1.0, 6
    freq = return_hist_factor(metric, value, timeframe)