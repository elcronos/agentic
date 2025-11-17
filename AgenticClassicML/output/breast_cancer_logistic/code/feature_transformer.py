
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from plexe.core.interfaces.feature_transformer import FeatureTransformer

class FeatureTransformerImplementation(FeatureTransformer):
    
    def transform(self, inputs: pd.DataFrame) -> pd.DataFrame:
        df = inputs.copy()
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        poly_df = pd.DataFrame(poly_features.fit_transform(df[['perimeter_worst', 'radius_worst', 'concave points_worst']]))
        
        poly_df.columns = [f'poly_{i}' for i in range(poly_df.shape[1])]
        df = pd.concat([df, poly_df], axis=1)
        
        df.drop(['perimeter_worst', 'radius_worst', 'concave points_worst'], axis=1, inplace=True)
        
        return df
