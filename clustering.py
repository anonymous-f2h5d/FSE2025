import json
import numpy as np
from scipy.interpolate import interp1d
from dtwWrapper import DtwWrapper

class Clustering:

    def __init__(self):
        """
            Initializes the class instance by loading and processing time series data from a JSON file.

            This constructor reads a JSON file (`timeseries_final_Major_Minor.json`) containing time series data
            for our repositories and their releases. It then extracts, processes, and stores the release time series,
            previous refactoring activity time series, and operation data for each release.

            Behavior:
            ---------
            1. Reads the `data/timeseries_final_Major_Minor.json` file to load time series data.
            2. For each repository and its releases:
               - The release time series and its associated metadata are stored.
               - The operations performed during the release and the accumulated history of time series data across
                 releases are stored.
            3. After processing the data, the `__interpolate()` method is called to interpolate the time series to a
            fixed length.

            Updates Attributes:
            -------------------
            release_time_series : np.ndarray
                An array of time series data for each release that has more than 3 data points.

            release_time_series_details : list
                A list of details for each release, including repository name, release version, creation date, and branch.

            operations_time_series : list
                A list of the operations performed during each release.

            history_time_series : np.ndarray
                An array of accumulated time series data representing the refactoring activities prior to each release.

            Notes:
            ------
            - The JSON file should have the following structure:
              ```
              {
                  "repository_name": {
                      "release_version": {
                          "timeseries": [...],
                          "operations": [...],
                          "created_at": "...",
                          "branch": "..."
                      },
                      ...
                  },
                  ...
              }
              ```
            - The `__interpolate()` method is automatically called to smooth and standardize the length of the time series.
        """
        with open("data/timeseries_final_Major_Minor.json") as json_file:
            time_series = json.load(json_file)

        # Defines the timeseries for each release
        release_time_series = []
        self.release_time_series_details = []
        operations_time_series = []
        # Defines the previous refactoring activities for each release
        history_time_series = []
        for repo, releases in time_series.items():
            # Ech repository
            accumulated_time_series = []
            for release, details in releases.items():
                # print(details)

                time_series = details['timeseries']
                operations = details['operations']
                created_at = details['created_at']
                branch = details['branch']

                if len(time_series) > 3:
                    # Holds the timeseries of the current version
                    self.release_time_series_details.append([repo, release, created_at, branch])
                    release_time_series.append(time_series)

                    operations_time_series.append(operations)
                    # Holds the timeseries of history
                    history_time_series.append(accumulated_time_series)

                accumulated_time_series = accumulated_time_series + time_series

        self.release_time_series = np.array(release_time_series, dtype=object)
        self.history_time_series = np.array(history_time_series, dtype=object)
        self.operations_time_series = operations_time_series

        self.__interpolate()

    def __interpolate(self, length=10):
        """
            Interpolates the time series data to a specified length.
            The data is interpolated to a fixed length, smoothing the data by filling in the missing values between
            points.

            Parameters:
            -----------
            length : int, optional (default=10)

            Behavior:
            ---------
            1. the method generates new x-coordinates spaced evenly over the original data, then linearly interpolates
            to the specified `length`, filling gaps in the series.

            Note:
            -----
            The `interp1d` function from `scipy.interpolate` is used for linear interpolation.
        """
        interpolated_list = []
        for input_list in self.release_time_series:
            # Create x-coordinates
            x = np.arange(len(input_list))
            f = interp1d(x, input_list, kind='linear')

            new_x = np.linspace(0, len(input_list) - 1, length)
            interpolated_data = f(new_x)
            interpolated_data_list = interpolated_data.tolist()
            interpolated_list.append(interpolated_data_list)
        self.interpolated_release_time_series = np.array(interpolated_list)

        interpolated_list = []
        length = int(length/10)
        for input_list in self.history_time_series:
            if len(input_list) > 0:
                x = np.arange(len(input_list))  # Create x-coordinates
                f = interp1d(x, input_list, kind='linear')

                new_x = np.linspace(0, len(input_list) - 1, length)
                interpolated_data = f(new_x)
                interpolated_data_list = interpolated_data.tolist()
                interpolated_list.append(interpolated_data_list)
            else:
                interpolated_list.append(np.zeros(length).tolist())
        self.interpolated_history_time_series = np.array(interpolated_list)


    def cluster_and_analyze(self, random_state=9):
        """
            Performs clustering on time series data using Dynamic Time Warping (DTW) and generates visualizations.

            This method utilizes the `DtwWrapper` class to apply Soft Dynamic Time Warping (SoftDTW) on the interpolated
            release and history time series. The clustering process groups the time series into `k` clusters and then
            visualizes the results using a box plot.

            Parameters:
            -----------
            random_state : int, optional (default=9)
                The seed value for random number generation to ensure reproducibility of the clustering results.
        """
        dtw_wrapper = DtwWrapper(self.interpolated_release_time_series,
                                 self.interpolated_history_time_series,
                                 DtwWrapper.softdtw,
                                 4,
                                 random_state,
                                 self.operations_time_series,
                                 self.release_time_series_details)

        dtw_wrapper.cluster()
        dtw_wrapper.box_plot()
        dtw_wrapper.radar_plots()
        dtw_wrapper.operations_distributions()

