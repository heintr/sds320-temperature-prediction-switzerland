# SDS320 gallery card

1. Figure: `figures/07_observed_predicted_residual_2024.png`
2. Title: Next-day temperature prediction across Switzerland
3. Research Q.: How accurately can next-day daily mean temperature across Switzerland be predicted from recent gridded weather, seasonality, spatial location, and elevation?
4. Summary: A lightweight machine-learning workflow predicts next-day mean temperature across sampled Swiss grid cells and evaluates how forecast skill varies through time, across space, and with elevation. The selected Histogram Gradient Boosting model reaches 1.567 °C MAE on the unseen 2024 test year, improving on persistence by 16.3%, and is additionally compared with an archived ECMWF TIGGE forecast benchmark.
5. Data: MeteoSwiss TabsD daily mean temperature and RhiresD daily precipitation, Swiss federal terrain elevation, and ECMWF TIGGE forecasts for an external benchmark.
6. Methods: Temporal train/validation/test split, feature engineering from recent weather and seasonality, linear regression and Histogram Gradient Boosting comparisons, persistence baseline, spatial residual/error analysis, permutation importance, and interactive Leafmap visualisation.
7. Keywords: temperature forecasting, Switzerland, MeteoSwiss, spatial regression, gradient boosting, elevation, Leafmap, ECMWF TIGGE
8. Lesson learned: Recent temperature carries most of the predictive signal, but terrain still matters: error generally increases with elevation, and spatial residual analysis reveals limitations that a single national MAE would hide.
9. GitHub link: <add-public-repository-url>
