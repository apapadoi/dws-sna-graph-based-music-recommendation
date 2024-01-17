import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.manifold import TSNE

with open('indexing_per_node.json', 'r') as file:
    indexing_per_node = json.load(file)
    
    optimized_embeddings = np.load('optimized_embeddings.npy')
    print(optimized_embeddings.shape)

    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(optimized_embeddings)

    print(embeddings_2d.shape)
    plt.figure(figsize=(12, 10))

    albums = [key.split('_')[2] for key in indexing_per_node.keys()]
    artists = [key.split('_')[1] for key in indexing_per_node.keys()]

    unique_albums = list(set(albums))
    unique_artists = list(set(artists))

    cmap = plt.get_cmap('viridis')  # You can use any colormap of your choice
    norm = Normalize(vmin=0, vmax=len(unique_artists)-1)
    scalar_map = ScalarMappable(cmap=cmap, norm=norm)

    desired_marker_size = len(unique_albums)
    available_markers = ['o', '^', 's', 'd', 'v', '<', '>', 'p', '*', 'h']
    repeat_count = (desired_marker_size - len(available_markers)) // len(available_markers) + 1
    available_markers = available_markers + available_markers * repeat_count
    available_markers = available_markers[:desired_marker_size]
    album_markers = {album: marker for album, marker in zip(unique_albums, available_markers)}

    total_albums = 0
    for album in unique_albums:
        # if total_albums >= 1000:
        #     break

        indices = [(key, idx) for key, idx in indexing_per_node.items() if key.split('_')[2] == album]
        marker = album_markers.get(album, 'o')
        color = scalar_map.to_rgba(unique_artists.index(indices[0][0].split('_')[1]))

        points_to_plot_x = []
        points_to_plot_y = []
        for key, idx in indices:
            
            points_to_plot_x.append(embeddings_2d[idx, 0])
            points_to_plot_y.append(embeddings_2d[idx, 1])

        plt.scatter(points_to_plot_x, points_to_plot_y, label=key.split('_')[1].replace('$', '\$'), marker=marker, color=color)
        total_albums += 1

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('TSNE of Embeddings')
    plt.legend(title='Artist')
    plt.grid(True)
    plt.show()