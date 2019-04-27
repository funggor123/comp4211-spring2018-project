import preprocess
import analysis
import feature

is_analysis = False

train_df, test_df = preprocess.load_data()
if is_analysis:
    analysis.analysis(train_df)
else:
    preprocess.preprocess(train_df)
feature.feature(train_df)

analysis.analysis_after_transform(train_df)



