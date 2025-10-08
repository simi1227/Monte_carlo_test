import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_japanese_candles(orig, perms):
    fig, ax = plt.subplots(figsize=(15, 6))
    x = mdates.date2num(orig.index)
    
    for i, xi in enumerate(x):
        o, h, l, c = orig.iloc[i][['open', 'high', 'low', 'close']]
        ax.vlines(xi, l, h, color='black', linewidth=1)
        ax.hlines(o, xi - 0.2, xi, color='black', linewidth=2)
        ax.hlines(c, xi, xi + 0.2, color='black', linewidth=2)

    for perm in perms:
        for i, xi in enumerate(x):
            o, h, l, c = perm.iloc[i][['open', 'high', 'low', 'close']]
            ax.vlines(xi, l, h, color='red', linewidth=1)
            ax.hlines(o, xi - 0.2, xi, color='red', linewidth=2)
            ax.hlines(c, xi, xi + 0.2, color='red', linewidth=2)

    ax.xaxis_date()
    fig.autofmt_xdate()
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Price")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def permutation_member(array):
    i = len(array)
    while i > 1:
        j = int(np.random.uniform(0, 1) * i) 
        if j >= i: j = i - 1
        i -= 1
        array[i], array[j] = array[j], array[i]
    return array

def permute_price(price, permute_index=None):
    if permute_index is None:
        permute_index = permutation_member(list(range(len(price) - 1)))
    log_prices = np.log(price)
    diff_logs = log_prices[1:] - log_prices[:-1]
    diff_perm = diff_logs[permute_index]
    cum_change = np.cumsum(diff_perm)
    new_log_prices = np.concatenate(([log_prices[0]], log_prices[0] + cum_change))
    new_prices = np.exp(new_log_prices)
    return new_prices

def permute_multi_prices(prices):
    assert all(len(price) == len(prices[0]) for price in prices)
    permute_index = permutation_member(list(range(len(prices[0]) - 1)))
    new_prices = [permute_price(price, permute_index=permute_index) for price in prices]
    return new_prices

def permute_bars(ohlcv, index_inter_bar=None, index_intra_bar=None):
    if not index_inter_bar:
        index_inter_bar = permutation_member(list(range(len(ohlcv) - 1)))
    if not index_intra_bar:
        index_intra_bar = permutation_member(list(range(len(ohlcv) - 2)))

    log_data = np.log(ohlcv)
    delta_h = log_data["high"].values - log_data["open"].values
    delta_l = log_data["low"].values - log_data["open"].values
    delta_c = log_data["close"].values - log_data["open"].values
    diff_deltas_h = np.concatenate((delta_h[1:-1][index_intra_bar], [delta_h[-1]]))
    diff_deltas_l = np.concatenate((delta_l[1:-1][index_intra_bar], [delta_l[-1]])) 
    diff_deltas_c = np.concatenate((delta_c[1:-1][index_intra_bar], [delta_c[-1]]))

    new_volumes = np.concatenate(
        (
            [log_data["volume"].values[0]], 
            log_data["volume"].values[1:-1][index_intra_bar], 
            [log_data["volume"].values[-1]]
        )
    )

    inter_open_to_close = log_data["open"].values[1:] - log_data["close"].values[:-1]
    diff_inter_open_to_close = inter_open_to_close[index_inter_bar]

    new_opens, new_highs, new_lows, new_closes = \
        [log_data["open"].values[0]], \
        [log_data["high"].values[0]], \
        [log_data["low"].values[0]], \
        [log_data["close"].values[0]]

    last_close = new_closes[0]
    for i_delta_h, i_delta_l, i_delta_c, inter_otc in zip(
        diff_deltas_h, diff_deltas_l, diff_deltas_c, diff_inter_open_to_close
    ):
        new_open = last_close + inter_otc
        new_high = new_open + i_delta_h
        new_low = new_open + i_delta_l
        new_close = new_open + i_delta_c
        new_opens.append(new_open)
        new_highs.append(new_high)
        new_lows.append(new_low)
        new_closes.append(new_close)
        last_close = new_close

    new_df = pd.DataFrame(
        {
            "open": new_opens,
            "high": new_highs,
            "low": new_lows,
            "close": new_closes,
            "volume": new_volumes
        }
    )
    new_df = np.exp(new_df)
    new_df.index = ohlcv.index
    return new_df

"""
Input:
    bars: list of pandas DataFrames, each containing OHLCV data for one instrument.
        Each DataFrame must:
            - Have columns: ["open", "high", "low", "close", "volume"]
            - Be indexed by datetime (or sortable time index)
            - Be sorted in ascending time order

    Assumptions:
        - bars[i] and bars[j] may have different date indices.
        - The function handles both:
            (i) Equal-length, aligned bars with shared time index
            (ii) Unequal-length bars with overlapping time windows
"""
from collections import defaultdict

def permute_multi_bars(bars):
    index_set = set(bars[0].index)
    if all(set(bar.index) == index_set for bar in bars):
        index_inter_bar = permutation_member(list(range(len(bars[0]) - 1)))
        index_intra_bar = permutation_member(list(range(len(bars[0]) - 2)))
        new_bars = [
            permute_bars(
                bar, 
                index_inter_bar=index_inter_bar, 
                index_intra_bar=index_intra_bar
            ) 
            for bar in bars
        ]
    else:
        bar_indices = list(range(len(bars)))
        index_to_dates = {k: set(list(bar.index)) for k, bar in zip(bar_indices, bars)}
        date_pool = set()
        for index in list(index_to_dates.values()):
            date_pool = date_pool.union(index)
        date_pool = list(date_pool)
        date_pool.sort()
        partitions, partition_idxs = [], []
        temp_partition = []
        temp_set = set([idx for idx, date_sets in index_to_dates.items() if date_pool[0] in date_sets])

        for i_date in date_pool:
            i_insts = set()
            for inst, date_sets in index_to_dates.items():
                if i_date in date_sets:
                    i_insts.add(inst)
            if i_insts == temp_set:
                temp_partition.append(i_date)
            else:
                partitions.append(temp_partition)
                partition_idxs.append(list(temp_set))
                temp_partition = [i_date]
                temp_set = i_insts
        partitions.append(temp_partition)
        partition_idxs.append(list(temp_set))

        chunked_bars = defaultdict(list)
        for partition, idx_list in zip(partitions, partition_idxs):
            permuted_bars = permute_multi_bars(
                [bars[idx].loc[partition] for idx in idx_list]
            )
            for idx, bar in zip(idx_list, permuted_bars):
                chunked_bars[idx].append(bar)

        new_bars = [None] * len(bars)
        for idx in bar_indices:
            new_bars[idx] = pd.concat(chunked_bars[idx], axis=0)
    return new_bars