from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from cams_downscaling.utils import get_db_connection, read_config


# MODIFY THE FOLLOWING VARIABLES ACCORDING TO YOUR NEEDS
# Iberia
iberia_2022_2023 = [1000, 1010, 1020, 1030, 1040, 1060, 1070, 1080, 1090, 1100]
iberia_2022 = [1001, 1011, 1021, 1031, 1041, 1061, 1071, 1081, 1091, 1101]
iberia_cams_2022_2023 = [-1000]

# Italy
italy_2022_2023 = [10001, 10101, 10201, 10301, 10401, 10601, 10701, 10801, 10901, 11001]
italy_2022 = [10011, 10111, 10211, 10311, 10411, 10611, 10711, 10811, 10911, 11011]
italy_cams_2022_2023 = [-10001]

# Poland
poland_2022 = [10012, 10112, 10212, 10312, 10412, 10612, 10712, 10812, 10912, 11012]
poland_cams_2022 = [-10002]

model_versions_to_plot = iberia_2022_2023 + iberia_2022 + iberia_cams_2022_2023 + italy_2022_2023 + italy_2022 + italy_cams_2022_2023 + poland_2022 + poland_cams_2022

# END OF VARIABLES TO MODIFY


config = read_config('/home/urbanaq/cams_downscaling/config')
DATA_PATH = Path(config['paths']['stations'])
OUTPUT_PATH = Path(config["paths"]["validation_results"]) / "diagrams"
CLUSTER_NAMES = config['places']['cluster_names']
POLLUTANTS = config['pollutants']['pollutants']


def plot_diagram(version, cursor):
    pollutant = POLLUTANTS[int(str(abs(version))[:1])] # Get pollutant from model version
    if len(str(abs(version))) == 4:
         query_clusters = f"""
                SELECT station_id, cluster
                FROM stations WHERE cluster LIKE '1%' OR cluster LIKE '2%';
                """
    elif int(str(abs(version))[4]) == 1:
        query_clusters = f"""
                SELECT station_id, cluster
                FROM stations WHERE cluster LIKE '4%';
                """
    else:
        query_clusters = f"""
                SELECT station_id, cluster
                FROM stations WHERE cluster LIKE '5%';
                """

    print(version)
    query = f"""
    SELECT station, bias, CRMSE, RMS_U
    FROM validation
    WHERE station != '90th' AND model_version = {version}
    """
    cursor.execute(query)
    data = cursor.fetchall()

    cursor.execute(query_clusters)
    stations = cursor.fetchall()
    # Get all different clusters
    clusters = list(set([s[1] for s in stations]))
    # Generate a random color for each cluster from a colormap with very different colors
    colors = plt.cm.get_cmap('tab20', len(clusters)).colors
    
    bias = []
    crmse = []
    rms_u = []
    color_points = []
    cluster_points = []

    for row in data:
        station = row[0]
        cluster = [s[1] for s in stations if s[0] == station]
        if not cluster:
            continue
        cluster = cluster[0]
        if cluster not in CLUSTER_NAMES:
            continue
        color_points.append(colors[clusters.index(cluster)])
        cluster_points.append(cluster)
        bias.append(row[1])
        crmse.append(row[2])
        rms_u.append(row[3])
    
    x_values = [crmse / (2 * rms_u) for crmse, rms_u in zip(crmse, rms_u)]
    y_values = [b / (2 * rms_u) for b, rms_u in zip(bias, rms_u)]

    plt.scatter(x_values, y_values, c=color_points, s=5)
    plt.xlabel('CRMSE / (2 * RMS_U)')
    plt.ylabel('bias / (2 * RMS_U)')

    # Add legend for each cluster
    for i, cluster in enumerate(list(set(cluster_points))):
        plt.scatter([], [], c=colors[i], label=CLUSTER_NAMES[cluster])
    plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5, title='Cluster', loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})

    if version > 0:
        plt.title(f'Target Diagram of Bias and CRMSE Normalized by RMS_U\nModel version: {version}')
    else:
        plt.title(f'Target Diagram of Bias and CRMSE Normalized by RMS_U\nCAMS Interpolated')

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    circle = patches.Circle((0, 0), 1, edgecolor='blue', facecolor='none', linestyle='--')
    plt.gca().add_patch(circle)
    
    output_path = OUTPUT_PATH / f'diagram_{version}.png'
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()

def main():
    conn = get_db_connection()
    
    cursor = conn.cursor()

    for version in model_versions_to_plot:
        plot_diagram(version, cursor)
        
    conn.close()

    return

if __name__ == "__main__":
    main()
