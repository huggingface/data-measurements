from datasets import Dataset

from data_measurements.measurements.cooccurences import Cooccurences, CooccurencesResults


def calc_p_word(word_count_df):
    word_count_df[PROP] = word_count_df[CNT] / float(sum(word_count_df[CNT]))
    vocab_counts_df = pd.DataFrame(
        word_count_df.sort_values(by=CNT, ascending=False))
    vocab_counts_df[VOCAB] = vocab_counts_df.index
    return vocab_counts_df


class PMIResults(CooccurencesResults):
    pass


class PMI(Cooccurences):
    name = "PMI"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def measure(self, dataset: Dataset) -> PMIResults:
        cooccurences = super().measure(dataset)

        # calc_p_word(word_count_df)
        print(self.vocab_counts_df["count"])
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
