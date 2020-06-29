import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from IPython.display import display, HTML
from sklearn.tree import tree, DecisionTreeClassifier, _tree


def pretty_print(df):
    #return display(HTML(df.to_html().replace("\\n", "<br>")).data)
    print (df.to_string())
def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
    inner_tree: _tree.Tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()

    def tree_dfs(node_id=0, current_rule=[]):
        # feature[i] holds the feature to split on, for the internal node i.
        split_feature = inner_tree.feature[node_id]
        if split_feature != _tree.TREE_UNDEFINED:  # internal node
            name = feature_names[split_feature]
            threshold = inner_tree.threshold[node_id]
            # left child
            left_rule = current_rule + ["({} <= {})".format(name, threshold)]
            tree_dfs(inner_tree.children_left[node_id], left_rule)
            # right child
            right_rule = current_rule + ["({} > {})".format(name, threshold)]
            tree_dfs(inner_tree.children_right[node_id], right_rule)
        else:  # leaf
            dist = inner_tree.value[node_id][0]
            dist = dist / dist.sum()
            max_idx = dist.argmax()
            if len(current_rule) == 0:
                rule_string = "ALL"
            else:
                rule_string = " and ".join(current_rule)
            # register new rule to dictionary
            selected_class = classes[max_idx]
            class_probability = dist[max_idx]
            class_rules = class_rules_dict.get(selected_class, [])
            class_rules.append((rule_string, class_probability))
            class_rules_dict[selected_class] = class_rules

    tree_dfs()  # start from root, node_id = 0
    return class_rules_dict


def cluster_report(data: pd.DataFrame, clusters, min_samples_leaf=50, pruning_level=0.01):
    # Create Model
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=pruning_level)
    tree.fit(data, clusters)

    # Generate Report
    feature_names = data.columns
    class_rule_dict = get_class_rules(tree, feature_names)

    report_class_list = []
    for class_name in class_rule_dict.keys():
        rule_list = class_rule_dict[class_name]
        combined_string = ""
        for rule in rule_list:
            combined_string += "[{}] {}\n\n".format(rule[1], rule[0])
        report_class_list.append((class_name, combined_string))

    cluster_instance_df = pd.Series(clusters).value_counts().reset_index()
    cluster_instance_df.columns = ['class_name', 'instance_count']
    report_df = pd.DataFrame(report_class_list, columns=['class_name', 'rule_list'])
    report_df = pd.merge(cluster_instance_df, report_df, on='class_name', how='left')
    pretty_print(report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']])

if __name__ == '__main__':
    data = pd.read_csv("../credit_card_data.csv")
    data.drop(data.columns[[0]], axis=1, inplace=True)
    missing = data.isna().sum()
    print(missing)

    data = data.fillna(data.median())
    print(data.isna().sum())

    # Let's assume we use all cols except CustomerID
    vals = data.iloc[:, 1:].values

    # Use the Elbow method to find a good number of clusters using WCSS
    # wcss = []
    # for ii in range(1, 30):
    #     kmeans = KMeans(n_clusters=ii, init="k-means++", n_init=10, max_iter=300)
    #     kmeans.fit_predict(vals)
    #     wcss.append(kmeans.inertia_)
    #
    # plt.plot(wcss, 'ro-', label="WCSS")
    # plt.title("Computing WCSS for KMeans++")
    # plt.xlabel("Number of clusters")
    # plt.ylabel("WCSS")
    # plt.show()

    best_cols = ["BALANCE", "PURCHASES", "CREDIT_LIMIT", "PAYMENTS", "PURCHASES_TRX", "MINIMUM_PAYMENTS"]
    # best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
    # iz lakta vidimo da je optimalan broj klastera oko 6
    kmeans = KMeans(n_clusters=8, init="k-means++", n_init=10, max_iter=300)
    best_vals = data[best_cols].iloc[:, 1:].values
    y_pred = kmeans.fit_predict(best_vals)

    data["cluster"] = y_pred
    best_cols.append("cluster")

    #sns.pairplot(data[best_cols], hue="cluster")

    cluster_report(data, y_pred, min_samples_leaf=50, pruning_level=0.001)
    plt.show()



