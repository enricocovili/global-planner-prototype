import csv
import numpy as np
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

def load_points_from_csv(file_path):
    points = []
    try:
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == "x":
                    continue
                x, y = map(float, row)
                points.append((x, y))
    except FileNotFoundError as e:
        print("file not found")
        return None
    return points

class Trajectory:
    def __init__(self, points: np.ndarray, is_loop: bool):
        self.points = points
        self.is_loop = is_loop
        self.update_length()
        
    @staticmethod
    def interpolate(a, b, t):
        return a + t * (b - a)

    def update_length(self):
        self.length = 0
        for i in range(1, len(self.points)):
            self.length += np.linalg.norm(self.points[i] - self.points[i - 1])
        if self.is_loop:
            self.length += np.linalg.norm(self.points[-1] - self.points[0])

    def do_resample(self, num_points):
        if self.points.size == 0 or num_points < 1:
            print("Invalid input data or num_points")
            return np.array([])
        
        arc_length = np.zeros(len(self.points))
        for i in range(1, len(self.points)):
            dist = np.linalg.norm(self.points[i] - self.points[i - 1])
            arc_length[i] = arc_length[i - 1] + dist
        
        closing_dist = np.linalg.norm(self.points[-1] - self.points[0])
        total_length = arc_length[-1] + closing_dist
        
        if total_length < 1e-8:
            return self.points
        
        new_arc_length = np.linspace(0, total_length, num_points)
        new_points = np.zeros((num_points, 2))
        
        for i, s in enumerate(new_arc_length):
            if s < arc_length[-1]:
                idx = np.searchsorted(arc_length, s)
                
                if idx == 0:
                    self.points[i] = self.points[0]
                else:
                    seg_length = arc_length[idx] - arc_length[idx - 1]
                    t = 0.0 if seg_length < 1e-8 else (s - arc_length[idx - 1]) / seg_length
                    new_points[i] = Trajectory.interpolate(self.points[idx - 1], self.points[idx], t)
            else:
                s_closing = s - arc_length[-1]
                t = 0.0 if closing_dist < 1e-8 else (s_closing / closing_dist)
                new_points[i] = Trajectory.interpolate(self.points[-1], self.points[0], t)

        self.points = new_points

    def do_smoothing(self):
        if self.points.size == 0:
            print("No points to smooth")
            return np.array([])

        # Apply a Savitzky-Golay filter for smoothing
        window_length = min(11, len(self.points) - (len(self.points) % 2 == 0))
        polyorder = 3

        smoothed_x = scipy.signal.savgol_filter(self.points[:, 0], window_length, polyorder)
        smoothed_y = scipy.signal.savgol_filter(self.points[:, 1], window_length, polyorder)

        self.points = np.vstack((smoothed_x, smoothed_y)).T

class Curvature:
    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory
        self.curvatures = np.array([])

    def segment_lengths(self) -> np.ndarray:
        """
        Compute segment lengths between points.
        """
        N = len(self.trajectory.points)
        ds = np.zeros(N)

        for i in range(N - 1):
            ds[i] = np.linalg.norm(self.trajectory.points[i + 1] - self.trajectory.points[i])

        if self.trajectory.is_loop:
            ds[N - 1] = np.linalg.norm(self.trajectory.points[0] - self.trajectory.points[N - 1])

        return ds

    def curvature(self, ds: np.ndarray, absolute: bool):
        """
        Compute curvature for a set of 2D points along a path.
        """
        def curv(fp: np.ndarray, fpp: np.ndarray) -> float:
            """Compute curvature given first and second derivatives."""
            if absolute:
                return np.abs((fp[0] * fpp[1] - fp[1] * fpp[0]) / (np.linalg.norm(fp) ** 3 + 1e-8))
            else:
                return (fp[0] * fpp[1] - fp[1] * fpp[0]) / (np.linalg.norm(fp) ** 3 + 1e-8)

        N = len(self.trajectory.points)
        k = np.zeros(N)

        if N < 3:
            return k

        fp_first = np.zeros(2)
        fp_last = np.zeros(2)

        fp_prev = np.zeros(2)
        fp = np.zeros(2)
        fp_next = (self.trajectory.points[0] - self.trajectory.points[2]) / (ds[0] + ds[1])

        # Compute curvature at i = 0
        if self.trajectory.is_loop:
            pred_idx = N - 1
            fp_prev = (self.trajectory.points[0] - self.trajectory.points[pred_idx - 1]) / (ds[pred_idx - 1] + ds[pred_idx])
            fp = (self.trajectory.points[1] - self.trajectory.points[pred_idx]) / (ds[pred_idx] + ds[0])
            fpp = (fp_next - fp_prev) / (ds[pred_idx] + ds[0])
            fp_first, fp_last = fp, fp_prev
        else:
            fp = (self.trajectory.points[1] - self.trajectory.points[0]) / ds[0]
            fpp = (fp_next - fp) / ds[0]

        k[0] = curv(fp, fpp)

        # Compute middle curvatures using central differences
        for i in range(1, N - 2):
            fp_prev = fp
            fp = fp_next
            fp_next = (self.trajectory.points[i + 2] - self.trajectory.points[i]) / (ds[i] + ds[i + 1])
            fpp = (fp_next - fp_prev) / (ds[i - 1] + ds[i])
            k[i] = curv(fp, fpp)

        # Compute curvatures for last two points
        curr_idx = N - 2
        fp_prev = fp
        fp = fp_next

        if self.trajectory.is_loop:
            fp_next = fp_last
        else:
            fp_next = (self.trajectory.points[curr_idx + 1] - self.trajectory.points[curr_idx]) / ds[curr_idx]

        fpp = (fp_next - fp_prev) / (ds[curr_idx] + ds[curr_idx - 1])
        k[curr_idx] = curv(fp, fpp)

        curr_idx += 1
        fp_prev = fp
        fp = fp_next

        if self.trajectory.is_loop:
            fpp = (fp_first - fp_prev) / (ds[curr_idx - 1] + ds[curr_idx])
        else:
            fpp = (fp_prev - fp) / ds[curr_idx - 1]

        k[curr_idx] = curv(fp, fpp)

        # Set curvatures at endpoints to zero
        k[0] = 0
        k[N - 1] = 0
        k[N - 2] = 0

        self.curvatures = k

    def butterworth_filter(self, cutoff_freq: float, order: int = 3) -> np.ndarray:
        """
        Applies a zero-phase Butterworth filter to smooth the curvature data.
        
        Parameters:
        - curvature: np.ndarray, curvature values at waypoints.
        - ds: np.ndarray, arc-length distances between waypoints.
        - cutoff_freq: float, cutoff frequency for filtering.
        - order: int, order of the Butterworth filter.

        Returns:
        - np.ndarray, smoothed curvature values.
        """
        # Design an analog Butterworth filter
        b, a = scipy.signal.butter(order, cutoff_freq, btype='low', analog=False)

        # Apply filtfilt for zero-phase filtering
        return scipy.signal.filtfilt(b, a, self.curvatures, method="gust")

    def do_computations(self, absolute: bool = False):
        """
        Perform curvature computations on a set of 2D points.
        
        Returns:
        - np.ndarray: Curvature values for each point.
        """
        ds = self.segment_lengths()
        curvs = self.curvature(ds, absolute)
        # apply butterworth filter
        curvs = self.butterworth_filter(0.1)
        self.curvatures = curvs

        return self.curvatures

def double_plot_curvature(curvatures: np.ndarray, track: np.ndarray):
    """
    Plot curvature values.
    
    Parameters:
    - curvatures (np.ndarray): Curvature values for each point.
    """
    
    import matplotlib.cm as cm

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # window title
    fig.canvas.set_window_title('Curvature Plot')

    colors = cm.rainbow(np.linspace(0, 1, len(curvatures)))
    ax1.scatter(range(len(curvatures)), curvatures, color=colors)
    ax1.set_ylabel('Curvature')
    ax1.set_title('Curvature')

    ax2.plot(track[:, 0], track[:, 1], color='black')
    ax2.scatter(track[:, 0], track[:, 1], color=colors)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Path with Curvature')

    plt.show()

def single_plot_curvature(curvatures: np.ndarray, track: np.ndarray):
    """
    Plot curvature values.
    
    Parameters:
    - curvatures (np.ndarray): Curvature values for each point.
    """
    
    import matplotlib.cm as cm

    # plot the track. the color of the single point is determined by the curvature
    # black = low curvature, red = high curvature

    # Normalize curvatures to range [0, 1] for colormap
    norm_curvatures = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures))
    colors = cm.RdYlBu(norm_curvatures)  # Red for high curvature, Blue for low curvature

    plt.scatter(track[:, 0], track[:, 1], color=colors)
    plt.plot(track[:, 0], track[:, 1], color='black')
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Path with Curvature')

    plt.show()

def segment_track(traj: Trajectory, curvature: np.ndarray):
    # divide the track in the following parts:
    # straights and curves.
    # use this strategy to divide the track:
    # if the abs(curvature) of a point is below 0.3 then it is straight

    straights = []
    curves = []

    for i in range(len(curvature)):
        if abs(curvature[i]) < 0.025:
            straights.append(traj.points[i])
        else:
            curves.append(traj.points[i])

    return np.array(straights), np.array(curves)


def plot_segments(straights: np.ndarray, curves: np.ndarray):
    
    """
    Plot the segmented track with straights and curves.
    
    Parameters:
    - straights: list of points representing straight segments.
    - curves: list of points representing curved segments.
    """
    

    plt.figure(figsize=(10, 8))

    if len(straights) > 0:
        plt.scatter(straights[:, 0], straights[:, 1], label='Straights', color='blue', marker='o')

    if len(curves) > 0:
        plt.scatter(curves[:, 0], curves[:, 1], label='Curves', color='red', marker='o')
    
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Segmented Track')
    plt.legend()
    plt.show()

def plot_curvatures_derivates(curvatures: np.ndarray):
    """
    Plot the curvature values and their derivatives.
    
    Parameters:
    - curvatures (np.ndarray): Curvature values for each point.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # window title
    fig.canvas.manager.set_window_title('Curvature Plot')

    ax1.plot(curvatures, color='blue')
    ax1.set_ylabel('Curvature')
    ax1.set_title('Curvature')

    # Compute first derivative of curvature
    d_curvatures = np.gradient(curvatures)

    ax2.plot(d_curvatures, color='red')
    ax2.set_ylabel('d(Curvature)/ds')
    ax2.set_title('Curvature Derivative')

    # Find local maxima and minima
    local_max_min = scipy.signal.argrelextrema(curvatures, np.greater)[0].tolist() + \
                    scipy.signal.argrelextrema(curvatures, np.less)[0].tolist()
    local_max_min.sort()

    for idx in local_max_min:
        ax1.axvline(x=idx, color='black', linestyle='--')
        ax2.axvline(x=idx, color='black', linestyle='--')

    plt.show()

# Example usage
if __name__ == "__main__":
    file_name = 'csvs/varano_autocross.csv'
    points = load_points_from_csv(file_name)
    if points is not None:
        points.append(points[0]) # close the loop
        # downsample points
        points = points[::1]
        # convert to numpy array
        points = np.array(points)
        traj = Trajectory(points, True)
        logging.info(f"Trajectory length: {traj.length:.3f}m")
        traj.do_resample(int(traj.length)) # points every ~1m
        traj.do_smoothing()
        curvature = Curvature(traj)
        curvature.do_computations(absolute=False)

        # single_plot_curvature(curvature.curvatures, traj.points)
        double_plot_curvature(curvature.curvatures, traj.points)
        # plot_curvatures_derivates(curvature.curvatures)
        straights, curves = segment_track(traj, curvature.curvatures)
        plot_segments(straights, curves)


