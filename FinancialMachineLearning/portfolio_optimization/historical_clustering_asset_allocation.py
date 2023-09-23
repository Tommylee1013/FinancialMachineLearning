import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from FinancialMachineLearning.portfolio_optimization.return_estimators import ReturnEstimation
from FinancialMachineLearning.portfolio_optimization.risk_metrics import RiskMetrics

class HierarchicalClusteringAssetAllocation:
    def __init__(self, calculate_expected_returns = 'mean'):
        self.weights = list()
        self.clusters = None
        self.ordered_indices = None
        self.returns_estimator = ReturnEstimation()
        self.risk_metrics = RiskMetrics()
        self.calculate_expected_returns = calculate_expected_returns

    @staticmethod
    def _compute_cluster_inertia(labels, asset_returns):
        unique_labels = np.unique(labels)
        inertia = [np.mean(pairwise_distances(asset_returns[:, labels == label])) for label in unique_labels]
        inertia = np.log(np.sum(inertia))
        return inertia

    def _get_optimal_number_of_clusters(self,
                                        correlation,
                                        asset_returns,
                                        num_reference_datasets=5,
                                        max_number_of_clusters=10):
        cluster_func = AgglomerativeClustering(affinity='precomputed', linkage='single')
        original_distance_matrix = np.sqrt(2 * (1 - correlation).round(5))
        gap_values = []
        for num_clusters in range(1, max_number_of_clusters + 1):
            cluster_func.n_clusters = num_clusters
            reference_inertias = []
            for _ in range(num_reference_datasets):
                reference_asset_returns = pd.DataFrame(np.random.rand(*asset_returns.shape))
                reference_correlation = np.array(reference_asset_returns.corr())
                reference_distance_matrix = np.sqrt(2 * (1 - reference_correlation).round(5))

                reference_cluster_assignments = cluster_func.fit_predict(reference_distance_matrix)
                inertia = self._compute_cluster_inertia(reference_cluster_assignments, reference_asset_returns.values)
                reference_inertias.append(inertia)
            expected_inertia = np.mean(reference_inertias)

            original_cluster_asignments = cluster_func.fit_predict(original_distance_matrix)
            inertia = self._compute_cluster_inertia(original_cluster_asignments, asset_returns.values)

            gap = expected_inertia - inertia
            gap_values.append(gap)

        return np.argmax(gap_values)

    @staticmethod
    def _tree_clustering(correlation, num_clusters):
        cluster_func = AgglomerativeClustering(n_clusters=num_clusters,
                                               affinity='precomputed',
                                               linkage='single')
        distance_matrix = np.sqrt(2 * (1 - correlation).round(5))
        cluster_func.fit(distance_matrix)
        return cluster_func.children_

    def _quasi_diagnalization(self, num_assets, curr_index):
        if curr_index < num_assets:
            return [curr_index]

        left = int(self.clusters[curr_index - num_assets, 0])
        right = int(self.clusters[curr_index - num_assets, 1])

        return (self._quasi_diagnalization(num_assets, left) + self._quasi_diagnalization(num_assets, right))

    @staticmethod
    def _get_inverse_variance_weights(covariance):
        inv_diag = 1 / np.diag(covariance.values)
        parity_w = inv_diag * (1 / np.sum(inv_diag))
        return parity_w

    def _get_cluster_variance(self, covariance, cluster_indices):
        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = self._get_inverse_variance_weights(cluster_covariance)
        cluster_variance = self.risk_metrics.calculate_variance(covariance=cluster_covariance, weights=parity_w)
        return cluster_variance

    def _get_cluster_sharpe_ratio(self, expected_asset_returns, covariance, cluster_indices):
        cluster_expected_returns = expected_asset_returns[cluster_indices]
        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = self._get_inverse_variance_weights(cluster_covariance)
        cluster_variance = self.risk_metrics.calculate_variance(covariance=cluster_covariance, weights=parity_w)
        cluster_sharpe_ratio = (parity_w @ cluster_expected_returns) / np.sqrt(cluster_variance)
        return cluster_sharpe_ratio

    def _get_cluster_expected_shortfall(self, asset_returns, covariance, confidence_level, cluster_indices):
        cluster_asset_returns = asset_returns.iloc[:, cluster_indices]
        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = self._get_inverse_variance_weights(cluster_covariance)
        portfolio_returns = cluster_asset_returns @ parity_w
        cluster_expected_shortfall = self.risk_metrics.calculate_expected_shortfall(returns=portfolio_returns,
                                                                                    confidence_level=confidence_level)
        return cluster_expected_shortfall

    def _get_cluster_conditional_drawdown_at_risk(self, asset_returns, covariance, confidence_level, cluster_indices):
        cluster_asset_returns = asset_returns.iloc[:, cluster_indices]
        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = self._get_inverse_variance_weights(cluster_covariance)
        portfolio_returns = cluster_asset_returns @ parity_w
        cluster_conditional_drawdown = self.risk_metrics.calculate_conditional_drawdown_risk(returns=portfolio_returns,
                                                                                             confidence_level=confidence_level)
        return cluster_conditional_drawdown

    def _recursive_bisection(self,
                             expected_asset_returns,
                             asset_returns,
                             covariance_matrix,
                             assets,
                             allocation_metric,
                             confidence_level):
        self.weights = pd.Series(1, index=self.ordered_indices)
        clustered_alphas = [self.ordered_indices]

        while clustered_alphas:
            clustered_alphas = [cluster[start:end]
                                for cluster in clustered_alphas
                                for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                                if len(cluster) > 1]

            for subcluster in range(0, len(clustered_alphas), 2):
                left_cluster = clustered_alphas[subcluster]
                right_cluster = clustered_alphas[subcluster + 1]

                if allocation_metric == 'minimum_variance':
                    left_cluster_variance = self._get_cluster_variance(covariance_matrix, left_cluster)
                    right_cluster_variance = self._get_cluster_variance(covariance_matrix, right_cluster)
                    alloc_factor = 1 - left_cluster_variance / (left_cluster_variance + right_cluster_variance)
                elif allocation_metric == 'minimum_standard_deviation':
                    left_cluster_sd = np.sqrt(self._get_cluster_variance(covariance_matrix, left_cluster))
                    right_cluster_sd = np.sqrt(self._get_cluster_variance(covariance_matrix, right_cluster))
                    alloc_factor = 1 - left_cluster_sd / (left_cluster_sd + right_cluster_sd)
                elif allocation_metric == 'sharpe_ratio':
                    left_cluster_sharpe_ratio = self._get_cluster_sharpe_ratio(expected_asset_returns,
                                                                               covariance_matrix,
                                                                               left_cluster)
                    right_cluster_sharpe_ratio = self._get_cluster_sharpe_ratio(expected_asset_returns,
                                                                                covariance_matrix,
                                                                                right_cluster)
                    alloc_factor = left_cluster_sharpe_ratio / (left_cluster_sharpe_ratio + right_cluster_sharpe_ratio)

                    if alloc_factor < 0 or alloc_factor > 1:
                        left_cluster_variance = self._get_cluster_variance(covariance_matrix, left_cluster)
                        right_cluster_variance = self._get_cluster_variance(covariance_matrix, right_cluster)
                        alloc_factor = 1 - left_cluster_variance / (left_cluster_variance + right_cluster_variance)
                elif allocation_metric == 'expected_shortfall':
                    left_cluster_expected_shortfall = self._get_cluster_expected_shortfall(asset_returns=asset_returns,
                                                                                           covariance=covariance_matrix,
                                                                                           confidence_level=confidence_level,
                                                                                           cluster_indices=left_cluster)
                    right_cluster_expected_shortfall = self._get_cluster_expected_shortfall(asset_returns=asset_returns,
                                                                                           covariance=covariance_matrix,
                                                                                           confidence_level=confidence_level,
                                                                                           cluster_indices=right_cluster)
                    alloc_factor = \
                        1 - left_cluster_expected_shortfall / (left_cluster_expected_shortfall + right_cluster_expected_shortfall)
                elif allocation_metric == 'conditional_drawdown_risk':
                    left_cluster_conditional_drawdown = self._get_cluster_conditional_drawdown_at_risk(asset_returns=asset_returns,
                                                         covariance=covariance_matrix,
                                                         confidence_level=confidence_level,
                                                         cluster_indices=left_cluster)
                    right_cluster_conditional_drawdown = self._get_cluster_conditional_drawdown_at_risk(asset_returns=asset_returns,
                                                         covariance=covariance_matrix,
                                                         confidence_level=confidence_level,
                                                         cluster_indices=right_cluster)
                    alloc_factor = \
                        1 - left_cluster_conditional_drawdown / (left_cluster_conditional_drawdown + right_cluster_conditional_drawdown)
                else:
                    alloc_factor = 0.5
                self.weights[left_cluster] *= alloc_factor
                self.weights[right_cluster] *= 1 - alloc_factor
        self.weights.index = assets[self.ordered_indices]
        self.weights = pd.DataFrame(self.weights)
        self.weights = self.weights.T

    @staticmethod
    def _cov2corr(covariance):
        d_matrix = np.zeros_like(covariance)
        diagnoal_sqrt = np.sqrt(np.diag(covariance))
        np.fill_diagonal(d_matrix, diagnoal_sqrt)
        d_inv = np.linalg.inv(d_matrix)
        corr = np.dot(np.dot(d_inv, covariance), d_inv)
        corr = pd.DataFrame(corr, index=covariance.columns, columns=covariance.columns)
        return corr

    @staticmethod
    def _perform_checks(asset_prices, asset_returns, covariance_matrix, allocation_metric):
        if asset_prices is None and asset_returns is None and covariance_matrix is None:
            raise ValueError("You need to supply either raw prices or returns or a covariance matrix of asset returns")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")

        if allocation_metric not in \
                {'minimum_variance', 'minimum_standard_deviation', 'sharpe_ratio',
                 'equal_weighting', 'expected_shortfall', 'conditional_drawdown_risk'}:
            raise ValueError("Unknown allocation metric specified. Supported metrics are - minimum_variance, "
                             "minimum_standard_deviation, sharpe_ratio, equal_weighting, expected_shortfall, "
                             "conditional_drawdown_risk")

    def allocate(self,
                 asset_names,
                 asset_prices=None,
                 asset_returns=None,
                 covariance_matrix=None,
                 expected_asset_returns=None,
                 allocation_metric='equal_weighting',
                 confidence_level=0.05,
                 optimal_num_clusters=None,
                 resample_by=None):
        self._perform_checks(asset_prices, asset_returns, covariance_matrix, allocation_metric)
        if allocation_metric == 'sharpe_ratio' and expected_asset_returns is None:
            if asset_prices is None:
                raise ValueError(
                    "Either provide pre-calculated expected returns or give raw asset prices for inbuilt returns calculation")

            if self.calculate_expected_returns == "mean":
                expected_asset_returns = self.returns_estimator.mean_historical_return_calculator(
                    asset_prices=asset_prices,
                    resample_by=resample_by)
            elif self.calculate_expected_returns == "exponential":
                expected_asset_returns = self.returns_estimator.exponential_historical_return_calculator(
                    asset_prices = asset_prices,
                    resample_by=resample_by)
            else:
                raise ValueError("Unknown returns specified. Supported returns - mean, exponential")

        if asset_returns is None:
            asset_returns = self.returns_estimator.return_calculator(asset_prices=asset_prices, resample_by=resample_by)
        asset_returns = pd.DataFrame(asset_returns, columns=asset_names)

        if covariance_matrix is None:
            covariance_matrix = asset_returns.cov()
        cov = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

        corr = self._cov2corr(covariance=cov)

        if not optimal_num_clusters:
            optimal_num_clusters = self._get_optimal_number_of_clusters(correlation=corr, asset_returns=asset_returns)

        self.clusters = self._tree_clustering(correlation=corr, num_clusters=optimal_num_clusters)

        num_assets = len(asset_names)
        self.ordered_indices = self._quasi_diagnalization(num_assets, 2 * num_assets - 2)

        self._recursive_bisection(expected_asset_returns=expected_asset_returns,
                                  asset_returns=asset_returns,
                                  covariance_matrix=cov,
                                  assets=asset_names,
                                  allocation_metric=allocation_metric,
                                  confidence_level=confidence_level)