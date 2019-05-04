from module.data_module import analyser, feature_engineer, preprocessor


is_analysis = False

train_df, test_df = preprocessor.load_data()
if is_analysis:
    analyser.analysis(train_df)
else:
    preprocessor.preprocess(train_df)

feature_engine = feature_engineer.FeatureEngine()
feature_engine.run(train_df)

analyser.analysis_after_transform(train_df)



