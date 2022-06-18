def main():
    import os
    import glob
    import yaml
    import pandas as pd

    cwd = os.getcwd()

    with open("config.yaml", 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)

    folders_data = os.listdir('data')
    file_data = {folder: max(glob.glob(os.path.join(cwd, 'data', folder, '*')), key=os.path.getctime) for folder in folders_data}

    df = None

    for file_key, file_value in file_data.items():
        try:
            df_temp = pd.read_csv(file_value, **config[file_key]['import'])
        except TypeError:
            df_temp = pd.read_csv(file_value)
        except UnicodeDecodeError:
            df_temp = pd.read_excel(file_value)
        
        if 'Bank' in config[file_key]['columns_old']:
            df_temp['Bank'] = file_key
        df_temp = df_temp[config[file_key]['columns_old']]
        df_temp = df_temp.rename(columns=dict(zip(config[file_key]['columns_old'], config[file_key]['columns_new'])))
        
        try:
            df_temp = df_temp[~df_temp['Expense name'].str.contains('|'.join(config[file_key]['Remove']))]
        except KeyError:
            pass

        df_temp['Day'] = pd.to_datetime(df_temp['Day'], format=config[file_key]['Day'])

        
        df = pd.concat([df, df_temp], ignore_index=True)
    df['Day'] = df['Day'].dt.round('D')

    df = df.sort_values(by=['Day', 'Expense name', 'Amount'])

    df['Amount'] = -df['Amount']

    filename = f"{df['Day'].max().strftime('%Y-%m-%d')}_{'-'.join(config.keys())}.xlsx"
    
    df['Day'] = df['Day'].dt.strftime('%d/%m/%Y')

    df.to_excel(os.path.join('output', filename), index=False)

if __name__ == '__main__':
    main()