# from random import sample
# from turtle import back
import pandas as pd
import shap


class explainer:
    def __init__(
        self, background_samples, samples_to_assess, clf, model_features, plot_len
    ):
        self.background_samples = background_samples
        self.samples_to_assess = samples_to_assess
        self.clf = clf
        self.model_features = model_features
        self.plot_len = plot_len

    def model_predict(self, data_asarray):
        data_asframe = pd.DataFrame(data_asarray, columns=self.model_features)
        return self.clf.predict_proba(data_asframe)[:, 1]

    def top_feat(self, shap_values):
        col_vals = pd.DataFrame(
            (pd.DataFrame(shap_values, columns=self.model_features).abs().mean())
        ).reset_index()
        col_vals.columns = ["col", "value"]
        col_vals.sort_values(by="value", ascending=False, inplace=True)
        return list(col_vals.head(self.plot_len)["col"])

    def exp(self):
        background_samples = self.background_samples[self.model_features]
        samp_main = self.samples_to_assess
        samples_to_assess = samp_main[self.model_features]
        shap_kernel_explainer = shap.KernelExplainer(
            self.model_predict, background_samples, link="identity"
        )
        shap_values = shap_kernel_explainer.shap_values(samples_to_assess)
        shap.summary_plot(
            shap_values, samples_to_assess, plot_type="dot", plot_size=[8, 6]
        )
        top_feat = self.top_feat(shap_values)
        for i in top_feat:
            shap.dependence_plot(
                i, shap_values, samples_to_assess, interaction_index=None
            )
            # Interaction index = none required due to matplotlib error - https://github.com/slundberg/shap/issues/2273
        self.shap_values = shap_values
        self.samp_main = samp_main
        self.samples_to_asses = samples_to_assess
        return shap_values, samp_main, samples_to_assess

    def individual_plotter(self):
        shap_values = self.shap_values
        samp_main = self.samp_main.reset_index()
        samples_to_assess = self.samples_to_asses
        shap_values_feat = pd.DataFrame(shap_values, columns=samples_to_assess.columns)
        features = self.model_features
        mergedss = pd.merge(
            left=samp_main,
            right=shap_values_feat,
            left_index=True,
            right_index=True,
            suffixes=["", "_tt"],
        )
        # for i in range(len(mergedss)):
        for i in range(min(10, len(samples_to_assess))):
            xy = mergedss.iloc[[i]]
            xy = xy.round(3)
            tr = xy[features].transpose().reset_index()
            tr.columns = ["feature", "feature_value"]
            match_details = (
                xy["Team"].values[0]
                + " vs "
                + xy["Opponent"].values[0]
                + ": "
                + xy["game_year"].values[0].astype("str")
                + " - round "
                + xy["round_num"].values[0].astype("str")
                + ".  Prediction = "
                + xy["score"].values[0].astype("str")
            )
            lister = [x + "_tt" for x in features]
            sh = xy[lister]
            sh.columns = features
            sh = sh.transpose().reset_index()
            sh.columns = ["feature", "shap_values"]
            sh["abs_shap_values"] = sh["shap_values"].abs()
            joiner = (
                pd.merge(left=tr, right=sh, on="feature")
                .sort_values(by="abs_shap_values", ascending=False)
                .head(10)
            )
            joiner["c_name"] = (
                joiner["feature"] + ":    " + joiner["feature_value"].astype("str")
            )
            joiner.sort_values(by="abs_shap_values", ascending=True, inplace=True)
            joiner[["c_name", "shap_values"]].plot.barh(
                x="c_name",
                y="shap_values",
                title=match_details,
                xlim=(-0.15, 0.15),
                figsize=(12, 3),
            )
