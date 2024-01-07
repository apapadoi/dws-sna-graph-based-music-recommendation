import json
from concurrent.futures import ThreadPoolExecutor
import time
import sys
import os
import gc
import random
import statistics
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import seaborn as sns
from sentence_transformers import SentenceTransformer, util


RANDOM_STATE = 42
YES_DATASET_TEST_LISTS_NUM = 10000
EMBEDDING_SIZE = 8
NEGATIVE_POSITIVE_SAMPLE_RATIO = 1
LEARNING_RATE = 0.01
NUM_EPOCHS = 1000
NUM_WORKERS = 12
NEGATIVE_CANDIDATE_SONGS = 10
K = 5
EARLY_STOPPING_PATIENCE = 10
NUM_OF_EVALUATION_ROUNDS = 10

random.seed(RANDOM_STATE)
model = SentenceTransformer('all-mpnet-base-v2')


if len(sys.argv) == 1:
    raise RuntimeError("Missing command line argument")
elif sys.argv[1] == 'yes':
    train_playlists = []
    test_playlists = []
    train_input_file = './yes_small/train.txt'
    test_input_file = './yes_small/test.txt'
    song_hash_file = './yes_small/song_hash.txt'
    song_per_integer_id = {}
    tags_hash_file = './yes_small/tag_hash.txt'
    tags_per_integer_id = {}
    tags_file = './yes_small/tags.txt'
    tags_per_song_integer_id = {}
    popularity_per_song = {}
    cooccurences_per_song_pair = {}
    playlist_lengths = []
    training_undirected_graph = nx.Graph()
    PLOT_DEGREE_DISTRIBUTION_TITLE = 'Yes.com Dataset - training only'
    PLOT_HIST_PLAYLIST_LENGTH_TITLE = 'Yes.come Dataset - both on training and testing'

    with open(song_hash_file, 'r') as file:
        for line in file.readlines():
            integer_id, song_title, artist = line.split('\t')
            song_per_integer_id[int(integer_id)] = f'{song_title.strip()} {artist.strip()}'

    assert len(list(song_per_integer_id.keys())) == 3168

    with open(tags_hash_file, 'r') as file:
        for line in file.readlines():
            integer_id, tag = line.split(',')
            tags_per_integer_id[int(integer_id)] = f'{tag.strip()}'

    assert len(list(tags_per_integer_id.keys())) == 250

    with open(tags_file, 'r') as file:
        for current_song_index, line in enumerate(file.readlines()):
            if '#' in line:
                tags_per_song_integer_id[current_song_index] = ''
                continue

            result = []
            for tag_integer_id in line.split(' '):
                result.append(tags_per_integer_id[int(tag_integer_id)])

            tags_per_song_integer_id[current_song_index] = ' '.join(result)

    assert len(list(tags_per_song_integer_id.keys())) == 3168

    with open(train_input_file, 'r') as file:
        lines = file.readlines()[2:]

        for train_playlist in lines:
            current_train_playlist_integer_ids = train_playlist.split(' ')[:-1]
            result = []
            for integer_id in current_train_playlist_integer_ids:
                result.append(f'{song_per_integer_id[int(integer_id)]}_{tags_per_song_integer_id[int(integer_id)]}')

            train_playlists.append(result)

    print(f'Total train playlists: {len(train_playlists)}')
    for current_train_playlist in train_playlists:
        playlist_lengths.append(len(current_train_playlist))

        for current_first_track_index, current_first_track in enumerate(current_train_playlist):
            for current_second_track_index, current_second_track in enumerate(current_train_playlist[current_first_track_index+1:]):
                if f'{current_first_track}_{current_second_track}' in cooccurences_per_song_pair:
                    cooccurences_per_song_pair[f'{current_first_track}_{current_second_track}'] += 1
                else:
                    cooccurences_per_song_pair[f'{current_first_track}_{current_second_track}'] = 1

        for current_first_track_index, current_first_track in enumerate(current_train_playlist):
            u = current_first_track
            if u in popularity_per_song:
                popularity_per_song[u] += 1
            else:
                popularity_per_song[u] = 1

            if current_first_track_index == len(current_train_playlist) - 1:
                break

            current_second_track = current_train_playlist[current_first_track_index + 1]

            v = current_second_track

            if training_undirected_graph.has_edge(u, v):
                training_undirected_graph[u][v]['weight'] += 1
            else:
                training_undirected_graph.add_edge(u, v, weight=1)

    with open(test_input_file, 'r') as file:
        lines = file.readlines()[2:]
        lines = random.sample(lines, YES_DATASET_TEST_LISTS_NUM)

        for test_playlist in lines:
            current_test_playlist_integer_ids = test_playlist.split(' ')[:-1]
            result = []
            for integer_id in current_test_playlist_integer_ids:
                result.append(f'{song_per_integer_id[int(integer_id)]}_{tags_per_song_integer_id[int(integer_id)]}')

            test_playlists.append(result)
            playlist_lengths.append(len(result))
elif sys.argv[1] == 'challenge':
    file_path = './challenge_set.json'
    PLOT_DEGREE_DISTRIBUTION_TITLE = 'Spotify Million Playlist Dataset Challenge - training only'
    PLOT_HIST_PLAYLIST_LENGTH_TITLE = 'Spotify Million Playlist Dataset Challenge - both on training and testing'

    with open(file_path, 'r') as file:
        data = json.load(file)

        del data['date']
        del data['version']
        del data['name']
        del data['description']

        playlists = data['playlists']

        print(f'Total playlists: {len(playlists)}')

        non_empty_playlists = []
        playlist_lengths = []

        for current_playlist in playlists:
            if len(current_playlist['tracks']) >= 3:
                non_empty_playlists.append(current_playlist)

        print(f'Non-empty playlists: {len(non_empty_playlists)}')
        
        train_playlists = []
        test_playlists = []

        for current_playlist in non_empty_playlists:
            playlist_lengths.append(len(current_playlist['tracks']))
            train_playlists.append(current_playlist['tracks'][:-1])
            test_playlists.append([current_playlist['tracks'][-1]])
            
            assert len(current_playlist['tracks']) == len(train_playlists[-1]) + len(test_playlists[-1]) and len(train_playlists[-1]) == len(current_playlist['tracks']) - 1

        training_multi_graph = nx.MultiGraph()

        counter = 0
        for current_train_playlist in train_playlists:
            for current_first_track_index, current_first_track in enumerate(current_train_playlist):
                if current_first_track_index == len(current_train_playlist) - 1:
                    break

                current_second_track = current_train_playlist[current_first_track_index + 1]
                if f"{current_first_track['track_name']}_{current_first_track['artist_name']}_{current_first_track['album_name']}" == 'Naive_The Kooks_Inside In / Inside Out' and f"{current_second_track['track_name']}_{current_second_track['artist_name']}_{current_second_track['album_name']}" == 'Mr. Brightside_The Killers_Inside In / Inside Out':
                    counter += 1
                training_multi_graph.add_edge(f"{current_first_track['track_name']}_{current_first_track['artist_name']}_{current_first_track['album_name']}", f"{current_second_track['track_name']}_{current_second_track['artist_name']}_{current_second_track['album_name']}")


        print(f'counter: {counter}')
        print(f'Number of nodes: {training_multi_graph.number_of_nodes()}')
        print(f'Number of edges: {training_multi_graph.number_of_edges()}')


        training_undirected_graph = nx.Graph()

        for u, v, data in training_multi_graph.edges(data=True):
            if training_undirected_graph.has_edge(u, v):
                training_undirected_graph[u][v]['weight'] += 1
            else:
                training_undirected_graph.add_edge(u, v, weight=1)
elif sys.argv[1] == 'spotify':
    PLOT_DEGREE_DISTRIBUTION_TITLE = 'Spotify Million Playlist Dataset - training only'
    PLOT_HIST_PLAYLIST_LENGTH_TITLE = 'Spotify Million Playlist Dataset - both on training and testing'
    TRAIN_PERCENTAGE_PER_FILE = 0.7
    folder_path = './spotify_million_playlist_dataset/data'

    def extract_x_y(file_name):
        parts = file_name.split('.')[2]
        x_y = parts.split('-')
        return int(x_y[0]), int(x_y[1])

    json_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.json')], key=extract_x_y)
    total_playlists = 0
    total_non_empty_playlists = 0
    test_playlists = []
    training_undirected_graph = nx.Graph()
    playlist_lengths = []
    popularity_per_song = {}
    cooccurences_per_song_pair = {}
    
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        if extract_x_y(file_name)[0] % 10000 == 0:
            print(file_name)
            gc.collect()

        if extract_x_y(file_name)[0] == 1000:
            break
            
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

            del data['info']

            playlists = data['playlists']

            total_playlists += len(playlists)

            for index, current_playlist in enumerate(playlists):
                if len(current_playlist['tracks']) >= 3:
                    playlist_lengths.append(len(current_playlist['tracks']))

                    if index < len(playlists) * TRAIN_PERCENTAGE_PER_FILE:
                        current_train_playlist = current_playlist['tracks']

                        for current_first_track_index, current_first_track in enumerate(current_train_playlist):
                            for current_second_track_index, current_second_track in enumerate(current_train_playlist[current_first_track_index+1:]):
                                if f"{current_first_track['track_name']} {current_first_track['artist_name']}_{current_first_track['album_name']}_{current_second_track['track_name']} {current_second_track['artist_name']}_{current_second_track['album_name']}" in cooccurences_per_song_pair:
                                    cooccurences_per_song_pair[f"{current_first_track['track_name']} {current_first_track['artist_name']}_{current_first_track['album_name']}_{current_second_track['track_name']} {current_second_track['artist_name']}_{current_second_track['album_name']}"] += 1
                                else:
                                    cooccurences_per_song_pair[f"{current_first_track['track_name']} {current_first_track['artist_name']}_{current_first_track['album_name']}_{current_second_track['track_name']} {current_second_track['artist_name']}_{current_second_track['album_name']}"] = 1

                        for current_first_track_index, current_first_track in enumerate(current_train_playlist):
                            u = f"{current_first_track['track_name']} {current_first_track['artist_name']}_{current_first_track['album_name']}"
                            if u in popularity_per_song:
                                popularity_per_song[u] += 1
                            else:
                                popularity_per_song[u] = 1

                            if current_first_track_index == len(current_train_playlist) - 1:
                                break

                            current_second_track = current_train_playlist[current_first_track_index + 1]
                                
                            v = f"{current_second_track['track_name']} {current_second_track['artist_name']}_{current_second_track['album_name']}"

                            if training_undirected_graph.has_edge(u, v):
                                training_undirected_graph[u][v]['weight'] += 1
                            else:
                                training_undirected_graph.add_edge(u, v, weight=1)
                    else:
                        test_playlists.append([f"{track['track_name']} {track['artist_name']}_{track['album_name']}" for track in current_playlist['tracks']])

                    total_non_empty_playlists += 1
            

    print(f'Total playlists: {total_playlists}')
    print(f'Non-empty playlists: {total_non_empty_playlists}')
else:
    raise RuntimeError("Invalid command line argument")

print(f'Number of nodes (undirected graph): {training_undirected_graph.number_of_nodes()}')
print(f'Number of edges (undirected graph): {training_undirected_graph.number_of_edges()}')

training_graph_nodes = list(training_undirected_graph.nodes())
degrees = sorted([degree for node, degree in training_undirected_graph.degree()], reverse=True)

plt.figure(figsize=(8, 6))
degree_value_counts = Counter(degrees)
x_values = list(degree_value_counts.keys())
y_values = list(degree_value_counts.values())
sns.scatterplot(x=x_values, y=y_values, alpha=0.75)
plt.xscale('log')
plt.yscale('log')  
plt.title(f'Log-log plot of degree distribution ({PLOT_DEGREE_DISTRIBUTION_TITLE})', fontsize=16)
plt.xlabel('Log-degree', fontsize=16)
plt.ylabel('Log-frequency', fontsize=16)
plt.grid(True)
plt.savefig("training_undirected_degree_distribution_plot.png")

plt.figure(figsize=(13, 8))
sns.histplot(playlist_lengths, binwidth=3)
plt.yscale('log')  
plt.title(f'Log plot of playlist length (based on #songs) distribution ({PLOT_HIST_PLAYLIST_LENGTH_TITLE})', fontsize=16)
plt.xlabel('Playlist length (based on #songs)', fontsize=16)
plt.ylabel('Log-frequency', fontsize=16)
plt.grid(True)
plt.savefig("training_undirected_playlist_length_distribution_plot.png")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {"cuda" if device.type == "cuda" else "cpu"}')

torch.manual_seed(RANDOM_STATE)

indexing_per_node = {}
non_neighbors_per_node = {}
neighbors_per_node = {}

rng = np.random.default_rng(seed=RANDOM_STATE)

def process_node(node, index):
    neighbors_result = list(training_undirected_graph.neighbors(node))
    non_neighbors = list(nx.non_neighbors(training_undirected_graph, node)) # TODO we can store initial non neighbors in another dict
    non_neighbors_result = random.sample(non_neighbors, len(neighbors_result) * NEGATIVE_POSITIVE_SAMPLE_RATIO)

    assert len(non_neighbors_result) == len(neighbors_result) * NEGATIVE_POSITIVE_SAMPLE_RATIO
    return node, index, non_neighbors_result, neighbors_result


with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_node, node, index): (node, index) for index, node in enumerate(training_graph_nodes)}
    for future in futures:
        node, indexing, non_neighbors, neighbors = future.result()
        indexing_per_node[node] = indexing
        non_neighbors_per_node[node] = non_neighbors
        neighbors_per_node[node] = neighbors


print("Indexing per node:", len(list(indexing_per_node.keys())))

with open('indexing_per_node.json', 'w') as file:
    json.dump(indexing_per_node, file)

print("Non-neighbors per node:", len(list(non_neighbors_per_node.keys())))
print("Neighbors per node:", len(list(neighbors_per_node.keys())))

embeddings = torch.randn(len(training_graph_nodes), EMBEDDING_SIZE, requires_grad=True, device=device)
print('embeddings shape before')
print(embeddings.shape)

start_time = time.time()

if sys.argv[2] == 'sentence-embedding':
    song_titles = [node.replace('_', ' ') for node in training_graph_nodes]
    song_sentence_embeddings = model.encode(song_titles, convert_to_tensor=True)

    cosine_scores = util.cos_sim(song_sentence_embeddings, song_sentence_embeddings)


def embedding_based_loss_function(embeddings, epoch):
    if epoch != 0:
        random.seed(epoch*100)
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(process_node, node, index): (node, index) for index, node in enumerate(training_graph_nodes)}
            for future in futures:
                node, indexing, non_neighbors, neighbors = future.result()
                non_neighbors_per_node[node] = non_neighbors

    loss = 0
    for node in training_graph_nodes: # TODO try to express as tensor multiplication too
        current_internal_sum = 0.0

        neighbor_tensor_list = []
        weight_tensor_list = []
        if sys.argv[2] == 'sentence-embedding':
            cosine_scores_tensor_list = []
            
        for neighbor in neighbors_per_node[node]:
            neighbor_tensor_list.append(embeddings[indexing_per_node[neighbor]])
            weight_tensor_list.append(torch.tensor(training_undirected_graph.get_edge_data(node, neighbor)['weight'], device=device))
            if sys.argv[2] == 'sentence-embedding':
                cosine_scores_tensor_list.append(cosine_scores[indexing_per_node[node], indexing_per_node[neighbor]])

        neighbor_tensor_list_stacked = torch.stack(neighbor_tensor_list)
        weight_tensor_list_stacked = torch.stack(weight_tensor_list)
        if sys.argv[2] == 'sentence-embedding':
            cosine_score_tensor_list_stacked = torch.stack(cosine_scores_tensor_list)

        expanded_node_tensor = embeddings[indexing_per_node[node]].expand_as(neighbor_tensor_list_stacked)
        if sys.argv[2] == 'sentence-embedding':
            lambda_i_j = torch.exp(cosine_score_tensor_list_stacked-torch.norm(expanded_node_tensor - neighbor_tensor_list_stacked, dim=1, p=2))
        else:
            lambda_i_j = torch.exp(-torch.norm(expanded_node_tensor - neighbor_tensor_list_stacked, dim=1, p=2))
        current_internal_sum = torch.sum(lambda_i_j - weight_tensor_list_stacked * torch.log10(lambda_i_j))

        non_neighbor_tensor_list = []
        if sys.argv[2] == 'sentence-embedding':
            non_neighbor_cosine_scores_tensor_list = []

        for non_neighbor in non_neighbors_per_node[node]:
            non_neighbor_tensor_list.append(embeddings[indexing_per_node[non_neighbor]])
            if sys.argv[2] == 'sentence-embedding':
                non_neighbor_cosine_scores_tensor_list.append(cosine_scores[indexing_per_node[node], indexing_per_node[non_neighbor]])

        non_neighbor_tensor_list_stacked = torch.stack(non_neighbor_tensor_list)
        if sys.argv[2] == 'sentence-embedding':
            non_neighbor_cosine_scores_tensor_list_stacked = torch.stack(non_neighbor_cosine_scores_tensor_list)
        
        if sys.argv[2] == 'sentence-embedding':
            lambda_i_j = torch.exp(non_neighbor_cosine_scores_tensor_list_stacked-torch.norm(expanded_node_tensor - non_neighbor_tensor_list_stacked, dim=1, p=2))
        else:
            lambda_i_j = torch.exp(-torch.norm(expanded_node_tensor - non_neighbor_tensor_list_stacked, dim=1, p=2))
        poisson_tensor_list = torch.poisson(lambda_i_j)

        current_internal_sum += torch.sum(lambda_i_j - poisson_tensor_list * torch.log10(lambda_i_j))

        loss += current_internal_sum

    print(f'loss: {loss}')
    
    return loss


optimizer = optim.Adam([embeddings], lr=LEARNING_RATE)
best_loss = float('inf')
early_stopping_counter = 0

for epoch in range(NUM_EPOCHS):
    print(f'epoch: {epoch+1}/{NUM_EPOCHS}')
    optimizer.zero_grad()
    loss = embedding_based_loss_function(embeddings, epoch)
    loss.backward()
    optimizer.step()

    if loss < best_loss:
        best_loss = loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping - Epoch: {epoch}, Best loss: {best_loss}')
            break

end_time = time.time()

execution_time = end_time - start_time
print(f"Total Training Time: {execution_time:.4f} seconds")

if device.type == 'cuda':
    optimized_embeddings = embeddings.cpu().detach().numpy()
else:
    optimized_embeddings = embeddings.detach().numpy()

np.save('optimized_embeddings.npy', optimized_embeddings)
print('embeddings shape after')
print(optimized_embeddings.shape)


# Evaluation
def process_test_playlist(current_test_playlist, current_test_playlist_index, evaluation_round):
    result_poisson = []
    result_popularity = []
    result_cooccurence = []
    if sys.argv[1] == 'yes':
        result_metadata = []

    print(f'test playlist {current_test_playlist_index+1}/{len(test_playlists)}')
    rng = np.random.default_rng(seed=current_test_playlist_index + (evaluation_round+1)*100)
    random.seed(current_test_playlist_index + (evaluation_round+1)*100)
    current_seen_tracks = []

    for current_first_track_index, current_first_track in enumerate(current_test_playlist):
        if current_first_track_index == len(current_test_playlist) - 1:
            break

        current_seen_tracks.append(current_first_track)

        current_second_track = current_test_playlist[current_first_track_index + 1]

        current_candidate_song_list = [current_second_track]

        for i in range(NEGATIVE_CANDIDATE_SONGS):
            random_index = rng.integers(0, len(training_graph_nodes))
            current_negative_candidate_track = training_graph_nodes[random_index]
            while current_negative_candidate_track in current_seen_tracks:
                random_index = rng.integers(0, len(training_graph_nodes))
                current_negative_candidate_track = training_graph_nodes[random_index]
            
            current_candidate_song_list.append(current_negative_candidate_track)


        if current_first_track not in training_graph_nodes:
            optimized_embedding_to_use_for_first_track = rng.standard_normal(EMBEDDING_SIZE)
        else:
            optimized_embedding_to_use_for_first_track = optimized_embeddings[indexing_per_node[current_first_track]]

        embeddings_for_candidate_song_list = []
        for track in current_candidate_song_list:
            if track in training_graph_nodes:
                embeddings_for_candidate_song_list.append((track, optimized_embeddings[indexing_per_node[track]]))
            else:
                embeddings_for_candidate_song_list.append((track, rng.standard_normal(EMBEDDING_SIZE)))

        def derive_correct_song_rank(distances_from_first_track_in_pair):
            indexed_distances = list(enumerate(distances_from_first_track_in_pair))

            sorted_distances = sorted(indexed_distances, key=lambda x: x[1])

            # Filter tuples that are either not in the training set or do not occur with the current song and have distance from current song equal to the candidate song set
            filtered_tuples = [tup for tup in indexed_distances if tup[1] == len(distances_from_first_track_in_pair)]
            if len(filtered_tuples) == len(distances_from_first_track_in_pair):
                correct_song_rank = rng.integers(1, len(distances_from_first_track_in_pair) + 1)
            else:
                random.shuffle(filtered_tuples)

                sorted_distances = [tup if tup[1] != len(distances_from_first_track_in_pair) else filtered_tuples.pop(0) for tup in sorted_distances]

                sorted_indices = [index for index, _ in sorted_distances]

                sorted_candidate_song_list = [embeddings_for_candidate_song_list[index] for index in sorted_indices]

                correct_song_rank = -1
                for current_candidate_song_index, song in enumerate(sorted_candidate_song_list):
                    if song[0] == current_second_track:
                        correct_song_rank = current_candidate_song_index + 1
                        break

            assert correct_song_rank > 0
            return correct_song_rank
        

        result_poisson.append(derive_correct_song_rank([np.linalg.norm(array[1] - optimized_embedding_to_use_for_first_track) for array in embeddings_for_candidate_song_list]))
        result_popularity.append(derive_correct_song_rank([-popularity_per_song[song] if song in popularity_per_song else len(current_candidate_song_list) for song in current_candidate_song_list]))
        result_cooccurence.append(derive_correct_song_rank([-cooccurences_per_song_pair[f'{current_first_track}_{song}'] if f'{current_first_track}_{song}' in cooccurences_per_song_pair else len(current_candidate_song_list) for song in current_candidate_song_list]))

        if sys.argv[1] == 'yes':
            def jaccard_index(tags1, tags2):
                set1 = set(tags1)
                set2 = set(tags2)
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                return intersection / union if union != 0 else len(current_candidate_song_list)

            distances_to_derive_rank = []
            for candidate_song in current_candidate_song_list:
                tags_for_candidate_song = candidate_song.split('_')[1].split(' ')
                tags_for_current_first_song = current_first_track.split('_')[1].split(' ')

                distances_to_derive_rank.append(jaccard_index(tags_for_current_first_song, tags_for_candidate_song))

            result_metadata.append(derive_correct_song_rank([-distance for distance in distances_to_derive_rank]))
            
    if sys.argv[1] == 'yes':
        return result_poisson, result_popularity, result_cooccurence, result_metadata
    else:
        return result_poisson, result_popularity, result_cooccurence


mrr_metrics_per_method_per_evaluation_round = {}
recall_at_k_metrics_per_method_per_evaluation_round = {}
for current_evaluation_round in range(NUM_OF_EVALUATION_ROUNDS):
    poisson_testing_ranks_per_transition = []
    popularity_testing_ranks_per_transition = []
    cooccurences_testing_ranks_per_transition = []
    if sys.argv[1] == 'yes':
        metadata_testing_ranks_per_transition = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_test_playlist, current_test_playlist, index, current_evaluation_round): (current_test_playlist, index) for index, current_test_playlist in enumerate(test_playlists)}
        for future in futures:
            if sys.argv[1] == 'yes':
                result_poisson, result_popularity, result_cooccurence, result_metadata = future.result()
                metadata_testing_ranks_per_transition.extend(result_metadata)
            else:
                result_poisson, result_popularity, result_cooccurence = future.result()

            poisson_testing_ranks_per_transition.extend(result_poisson)
            popularity_testing_ranks_per_transition.extend(result_popularity)
            cooccurences_testing_ranks_per_transition.extend(result_cooccurence)

    assert len(poisson_testing_ranks_per_transition) == len(popularity_testing_ranks_per_transition) == len(cooccurences_testing_ranks_per_transition)
    if sys.argv[1] == 'yes':
        assert len(metadata_testing_ranks_per_transition) == len(poisson_testing_ranks_per_transition)
    
    def calculate_metrics_per_method(method, ranks):
        reciprocal_ranks = [1 / rank for rank in ranks if rank != 0]
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

        print(f"Method: {method} - Mean Reciprocal Rank (MRR): {mrr:.3f}")

        relevant_items = sum(1 for rank in ranks if rank <= K)
        total_transitions = len(ranks)

        recall_at_k = relevant_items / total_transitions if total_transitions != 0 else 0.0

        print(f"Method: {method} - Recall@{K}: {recall_at_k:.3f}")
        return mrr, recall_at_k

    print(f'Round {current_evaluation_round+1}/{NUM_OF_EVALUATION_ROUNDS} results:')

    current_popularity_mrr, current_popularity_recall_at_k = calculate_metrics_per_method('Popularity', popularity_testing_ranks_per_transition)
    if 'Popularity' in mrr_metrics_per_method_per_evaluation_round:
        mrr_metrics_per_method_per_evaluation_round['Popularity'].append(current_popularity_mrr)
    else:
        mrr_metrics_per_method_per_evaluation_round['Popularity'] = [current_popularity_mrr]

    if 'Popularity' in recall_at_k_metrics_per_method_per_evaluation_round:
        recall_at_k_metrics_per_method_per_evaluation_round['Popularity'].append(current_popularity_recall_at_k)
    else:
        recall_at_k_metrics_per_method_per_evaluation_round['Popularity'] = [current_popularity_recall_at_k]

    current_cooccurrence_mrr, current_cooccurrence_recall_at_k = calculate_metrics_per_method('Co-occurence', cooccurences_testing_ranks_per_transition)
    if 'Co-occurence' in mrr_metrics_per_method_per_evaluation_round:
        mrr_metrics_per_method_per_evaluation_round['Co-occurence'].append(current_cooccurrence_mrr)
    else:
        mrr_metrics_per_method_per_evaluation_round['Co-occurence'] = [current_cooccurrence_mrr]

    if 'Co-occurence' in recall_at_k_metrics_per_method_per_evaluation_round:
        recall_at_k_metrics_per_method_per_evaluation_round['Co-occurence'].append(current_cooccurrence_recall_at_k)
    else:
        recall_at_k_metrics_per_method_per_evaluation_round['Co-occurence'] = [current_cooccurrence_recall_at_k]


    if sys.argv[1] == 'yes':
        current_metadata_mrr, current_metadata_recall_at_k = calculate_metrics_per_method('Meta-data', metadata_testing_ranks_per_transition)
        if 'Meta-data' in mrr_metrics_per_method_per_evaluation_round:
            mrr_metrics_per_method_per_evaluation_round['Meta-data'].append(current_metadata_mrr)
        else:
            mrr_metrics_per_method_per_evaluation_round['Meta-data'] = [current_metadata_mrr]

        if 'Meta-data' in recall_at_k_metrics_per_method_per_evaluation_round:
            recall_at_k_metrics_per_method_per_evaluation_round['Meta-data'].append(current_metadata_recall_at_k)
        else:
            recall_at_k_metrics_per_method_per_evaluation_round['Meta-data'] = [current_metadata_recall_at_k]
    
    if sys.argv[2] == 'sentence-embedding':
        current_poisson_mrr, current_poisson_recall_at_k = calculate_metrics_per_method('SentencePoissonEmb', poisson_testing_ranks_per_transition)
        name_to_use_for_storing_metrics = 'SentencePoissonEmb'
    else:
        current_poisson_mrr, current_poisson_recall_at_k = calculate_metrics_per_method('PoissonEmb', poisson_testing_ranks_per_transition)
        name_to_use_for_storing_metrics = 'PoissonEmb'

    if name_to_use_for_storing_metrics in mrr_metrics_per_method_per_evaluation_round:
        mrr_metrics_per_method_per_evaluation_round[name_to_use_for_storing_metrics].append(current_poisson_mrr)
    else:
        mrr_metrics_per_method_per_evaluation_round[name_to_use_for_storing_metrics] = [current_poisson_mrr]

    if name_to_use_for_storing_metrics in recall_at_k_metrics_per_method_per_evaluation_round:
        recall_at_k_metrics_per_method_per_evaluation_round[name_to_use_for_storing_metrics].append(current_poisson_recall_at_k)
    else:
        recall_at_k_metrics_per_method_per_evaluation_round[name_to_use_for_storing_metrics] = [current_poisson_recall_at_k]


for key in mrr_metrics_per_method_per_evaluation_round.keys():
    assert len(mrr_metrics_per_method_per_evaluation_round[key]) == NUM_OF_EVALUATION_ROUNDS
    assert len(recall_at_k_metrics_per_method_per_evaluation_round[key]) == NUM_OF_EVALUATION_ROUNDS


print(f'\nResults across {NUM_OF_EVALUATION_ROUNDS} evaluation rounds')
mrr_evaluation_results = {}
for key, values in mrr_metrics_per_method_per_evaluation_round.items():
    mean = statistics.mean(values)
    variance = statistics.stdev(values)
    print(f"Method: {key} - Mean Reciprocal Rank (MRR): {mean:.3f} +- {variance:.3f}")

recall_at_k_evaluation_results = {}
for key, values in recall_at_k_metrics_per_method_per_evaluation_round.items():
    mean = statistics.mean(values)
    variance = statistics.stdev(values)
    print(f"Method: {key} - Recall@{K}: {mean:.3f} +- {variance:.3f}")

# future TODO try yes complete and yes big or not sample test playlists
# future TODO pass euclidean distance through sigmoid
# future TODO state that the yes dataset assumes that all testing set songs are found in the training set