from data_measurements.measurements.cooccurences import (
    Cooccurences,
    CooccurencesResults
)

from datasets import Dataset


class PMIResults(CooccurencesResults):
    pass


class PMI(Cooccurences):
    name = "PMI"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def measure(self, dataset: Dataset) -> CooccurencesResults:
        pass
        # cooccurences = super().measure(dataset)
        # print(self.vocab_counts_df["count"])
        # # Calculation of p(subgroup)
        # subgroup_prob = self.vocab_counts_df.loc[subgroup]["proportion"]
        # # Calculation of p(subgroup|word) = count(subgroup,word) / count(word)
        # # Because the indices match (the vocab words),
        # # this division doesn't need to specify the index (I think?!)
        # vocab_cooc_df.columns = ["cooc"]
        # p_subgroup_g_word = (
        #         vocab_cooc_df["cooc"] / self.vocab_counts_df["count"])
        # logs.info("p_subgroup_g_word is")
        # logs.info(p_subgroup_g_word)
        # pmi_df = pd.DataFrame()
        # pmi_df[subgroup] = np.log(p_subgroup_g_word / subgroup_prob).dropna()

