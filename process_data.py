import pandas as pd

def main(float_features, data_file, new_data_file):
    data = pd.read_csv(data_file)
    for feature in float_features:
        min_v = data[feature].min()
        max_v = data[feature].max()
        data[feature] = (data[feature] - min_v)/(max_v - min_v)
    data.to_csv(new_data_file)


if __name__ == "__main__":
    float_feature = ['IAT[t]', 'IAT[t-1]', 'BD[t]', 'BD[t-1]']
    main(float_feature, '/Users/Jean/Downloads/model2.csv', '/Users/Jean/Downloads/model2_new.csv')

    float_feature += ['RiskPremium1', 'SMB1', 'HML1', 'RMW1', 'CMA1', 'Rf']
    main(float_feature, '/Users/Jean/Downloads/model3.csv', '/Users/Jean/Downloads/model3_new.csv')

    float_feature.append("R[t-1]")
    main(float_feature, '/Users/Jean/Downloads/model4.csv', '/Users/Jean/Downloads/model4_new.csv')