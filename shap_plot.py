###
### after automl modeling, plot with shap tools
###

def getColumnN(df, binsize, mass0):
    columns = []
    for i in range(df.shape[1]):
        columns.append('m/z %s' %(int(binsize*i + mass0)))
    return columns
  

# shap)values
save_model = load_model(pathtofile)
train_pipe = save_model[:-1].transform(ms_df)
explainer = shap.TreeExplainer(save_model.named_steps["trained_model"])
shap_values = explainer.shap_values(train_pipe)

shap.summary_plot(shap_values, max_display=10, 
                  class_names=['class', 'names', 'etc'], 
                  feature_names=getColumnN(ms_df))
