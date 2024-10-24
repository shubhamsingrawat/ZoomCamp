from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    imputer = SimpleImputer(strategy = 'mean')
    data['bmi']=imputer.fit_transform(data[['bmi']])
    encoded_data= data.copy()

    features_to_scale=['age','bmi']
    scaler = MinMaxScaler()
    encoded_data[features_to_scale]=scaler.fit_transform(encoded_data[features_to_scale])
    scaler = QuantileTransformer(output_distribution='uniform')
    encoded_data['avg_glucose_level'] = scaler.fit_transform(encoded_data[['avg_glucose_level']])

    df = encoded_data.copy()

    columns_to_encode = ['Residence_type', 'work_type', 'smoking_status','ever_married','gender']
    for column in columns_to_encode:
        encoded_column = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, encoded_column], axis=1)
        df = df.drop(columns=[column],axis=1)

    df = df.astype(int)
    df.drop('id',axis=1,inplace=True)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'