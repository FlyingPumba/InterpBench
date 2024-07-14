import pandas as pd


def stats_to_df(cache_dict, column_names) -> pd.DataFrame:
    column_names = ["name"] + column_names
    df = pd.DataFrame(columns=column_names)
    for k, v in cache_dict.items():
        entry = {
            "name": (
                (k.name + f", head {str(k.index).split(',')[2]}")
                if "attn" in k.name
                else k.name
            ),
        }
        entry.update(
            {column: v[column] for column in column_names if column in v.keys()}
        )
        df = pd.concat([df, pd.DataFrame(entry, index=[0])], axis=0, ignore_index=True)
    return df