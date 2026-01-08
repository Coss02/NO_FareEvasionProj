import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt

class TransportNetworkAnalyzer:
    """
    Classe per analizzare reti di trasporto pubblico e calcolare la matrice delle distanze.
    """
    
    def __init__(self, fermate_file, archi_file, rischio_file, id_column='CODICE_PAL'):
        """
        Inizializza l'analizzatore caricando i file shapefile e il file di rischio.
        
        Args:
            fermate_file: percorso al file shapefile delle fermate (punti)
            archi_file: percorso al file shapefile degli archi (polyline)
            rischio_file: percorso al file Excel contenente i valori di rischio per gli archi
            id_column: nome della colonna da usare come ID delle fermate
                      Se None, usa 'codice_pal'
        """
        print("Caricamento dei dati...")
        self.fermate = gpd.read_file(fermate_file)
        self.archi = gpd.read_file(archi_file)
        self.rischio_df = pd.read_excel(rischio_file)  # Carica il file di rischio
        self.graph = None
        self.distance_matrix = None
        self.node_names = None
        self.id_column = id_column
        
        print(f"✓ Caricate {len(self.fermate)} fermate")
        print(f"✓ Caricati {len(self.archi)} archi")
        print(f"✓ Colonna ID fermate: '{self.id_column}'")
        
    def explore_data(self):
        """Esplora la struttura dei dati caricati."""
        print("\n" + "="*60)
        print("STRUTTURA DATI FERMATE")
        print("="*60)
        print(f"\nColonne disponibili: {list(self.fermate.columns)}")
        print(f"\nPrime righe:")
        print(self.fermate.head())
        
        print("\n" + "="*60)
        print("STRUTTURA DATI ARCHI")
        print("="*60)
        print(f"\nColonne disponibili: {list(self.archi.columns)}")
        print(f"\nPrime righe:")
        print(self.archi.head())
    
    def build_graph(self, weight_column=None):
        """
        Costruisce un grafo NetworkX dalla rete di trasporto.
        
        Args:
            weight_column: nome della colonna da usare come peso (es. lunghezza, tempo)
                          Se None, usa il valore di rischio
        """
        print("\nCostruzione del grafo...")
        self.graph = nx.Graph()
        
        # Crea un mapping da indice a ID fermata
        fermata_id_map = {}
        
        # Aggiungi le fermate come nodi usando la colonna 'codice_pal'
        for idx, fermata in self.fermate.iterrows():
            node_id = fermata[self.id_column]  # Usa 'codice_pal' come ID
            
            fermata_id_map[idx] = node_id
            
            self.graph.add_node(
                node_id,
                pos=(fermata.geometry.x, fermata.geometry.y),
                **fermata.to_dict()
            )
        
        # Aggiungi gli archi
        edges_added = 0
        for idx, arco in self.archi.iterrows():
            # Trova le fermate più vicine agli estremi della linea
            line = arco.geometry
            
            if isinstance(line, LineString):
                # Primo punto della linea
                start_point = Point(line.coords[0])
                # Ultimo punto della linea
                end_point = Point(line.coords[-1])
                
                # Trova la fermata più vicina al punto iniziale
                start_distances = self.fermate.geometry.distance(start_point)
                start_idx = start_distances.idxmin()
                start_node = fermata_id_map[start_idx]
                
                # Trova la fermata più vicina al punto finale
                end_distances = self.fermate.geometry.distance(end_point)
                end_idx = end_distances.idxmin()
                end_node = fermata_id_map[end_idx]
                # Calcola il rischio
                arco_id = arco['PIPPO']  # Usa 'PIPPO' come ID dell'arco
                print(f"ID arco corrente: {arco_id}")

                # Verifica se l'ID esiste nel DataFrame
                rischio_rows = self.rischio_df[self.rischio_df['Arco'] == arco_id]

                # Controlla se ci sono risultati e recupera il valore di rischio
                if not rischio_rows.empty:
                    rischio = rischio_rows['Somma di Rischio'].values
                    weight = rischio[0]  # Usa il valore di rischio
                else:
                    # Se non trovato, usa un valore di default (ad esempio, 1)
                    print(f"⚠ Nessun rischio trovato per l'arco {arco_id}, uso peso di default")
                    weight = 1
                
                # Aggiungi l'arco
                if start_node != end_node:
                    self.graph.add_edge(start_node, end_node, weight=weight, **arco.to_dict())
                    edges_added += 1
        
        print(f"✓ Grafo costruito: {self.graph.number_of_nodes()} nodi, {self.graph.number_of_edges()} archi")
        
        # Verifica connettività
        if nx.is_connected(self.graph):
            print("✓ Il grafo è connesso (tutte le fermate sono raggiungibili)")
        else:
            components = list(nx.connected_components(self.graph))
            print(f"⚠ Il grafo NON è connesso: {len(components)} componenti separate")
            print(f"  Componente più grande: {len(max(components, key=len))} nodi")
    
    def calculate_distance_matrix(self, method='dijkstra'):
        """
        Calcola la matrice delle distanze tra tutte le coppie di nodi.
        
        Args:
            method: 'dijkstra' o 'floyd_warshall'
        
        Returns:
            DataFrame con la matrice delle distanze
        """
        if self.graph is None:
            raise ValueError("Prima devi costruire il grafo con build_graph()")
        
        print(f"\nCalcolo matrice delle distanze (metodo: {method})...")
        n_nodes = self.graph.number_of_nodes()
        
        if method == 'floyd_warshall':
            # Floyd-Warshall: calcola tutti i cammini minimi contemporaneamente
            distances = dict(nx.floyd_warshall(self.graph, weight='weight'))
        else:
            # Dijkstra: calcola i cammini minimi da ogni nodo
            distances = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))
        
        # Converti in matrice
        nodes = sorted(self.graph.nodes())
        matrix = np.zeros((n_nodes, n_nodes))
        
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if node_j in distances[node_i]:
                    matrix[i][j] = distances[node_i][node_j]
                else:
                    # Nodi non connessi
                    matrix[i][j] = np.inf
        
        # Crea DataFrame per una visualizzazione migliore
        self.node_names = nodes
        self.distance_matrix = pd.DataFrame(
            matrix,
            index=nodes,
            columns=nodes
        )
        
        print(f"✓ Matrice delle distanze calcolata: {n_nodes}x{n_nodes}")
        return self.distance_matrix
    
    def get_statistics(self):
        """Calcola statistiche sulla matrice delle distanze."""
        if self.distance_matrix is None:
            raise ValueError("Prima devi calcolare la matrice con calculate_distance_matrix()")
        
        # Rimuovi la diagonale (distanza da un nodo a se stesso)
        distances = self.distance_matrix.values[np.triu_indices_from(self.distance_matrix, k=1)]
        # Rimuovi infiniti (nodi non connessi)
        distances = distances[np.isfinite(distances)]
        
        stats = {
            'min': np.min(distances),
            'max': np.max(distances),
            'mean': np.mean(distances),
            'median': np.median(distances),
            'std': np.std(distances)
        }
        
        print("\n" + "="*60)
        print("STATISTICHE MATRICE DELLE DISTANZE")
        print("="*60)
        print(f"Distanza minima:  {stats['min']:.2f}")
        print(f"Distanza massima: {stats['max']:.2f}")
        print(f"Distanza media:   {stats['mean']:.2f}")
        print(f"Distanza mediana: {stats['median']:.2f}")
        print(f"Deviazione std:   {stats['std']:.2f}")
        
        return stats
    
    def find_shortest_path(self, source, target):
        """
        Trova il percorso più breve tra due nodi.
        
        Args:
            source: ID nodo di partenza
            target: ID nodo di arrivo
            
        Returns:
            tuple: (percorso, distanza)
        """
        if self.graph is None:
            raise ValueError("Prima devi costruire il grafo con build_graph()")
        
        try:
            path = nx.shortest_path(self.graph, source, target, weight='weight')
            length = nx.shortest_path_length(self.graph, source, target, weight='weight')
            return path, length
        except nx.NetworkXNoPath:
            return None, np.inf
    
    def visualize_network(self, figsize=(15, 10)):
        """Visualizza la rete di trasporto."""
        if self.graph is None:
            raise ValueError("Prima devi costruire il grafo con build_graph()")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Disegna gli archi (strade)
        self.archi.plot(ax=ax, color='lightblue', linewidth=1, alpha=0.6)
        
        # Disegna le fermate
        self.fermate.plot(ax=ax, color='red', markersize=20, alpha=0.7)
        
        ax.set_title('Rete di Trasporto Pubblico', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitudine')
        ax.set_ylabel('Latitudine')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def export_distance_matrix(self, output_file):
        """
        Esporta la matrice delle distanze in un file CSV.
        
        Args:
            output_file: percorso del file di output
        """
        if self.distance_matrix is None:
            raise ValueError("Prima devi calcolare la matrice con calculate_distance_matrix()")
        
        self.distance_matrix.to_csv(output_file)
        print(f"\n✓ Matrice esportata in: {output_file}")


# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    # Percorsi ai file (modifica con i tuoi percorsi)
    FERMATE_FILE = "Cartografia base/Fermate/00_FERMATE_2015_05_12_point.shp"
    ARCHI_FILE = "Cartografia base/Archi fermate/01_ARCHI_FERMATE_2015_05_12_polyline.shp"
    RISCHIO_FILE = "Rischio.xlsx"
    
    # Crea l'analizzatore
    analyzer = TransportNetworkAnalyzer(FERMATE_FILE, ARCHI_FILE, RISCHIO_FILE)
    
    # Esplora i dati (opzionale, per capire la struttura)
    analyzer.explore_data()
    
    # Costruisci il grafo
    analyzer.build_graph()
    
    # Calcola la matrice delle distanze
    distance_matrix = analyzer.calculate_distance_matrix(method='dijkstra')
    
    # Mostra statistiche
    stats = analyzer.get_statistics()
    
    # Visualizza parte della matrice
    print("\n" + "="*60)
    print("ESEMPIO MATRICE DELLE DISTANZE (prime 5x5 fermate)")
    print("="*60)
    print(distance_matrix.iloc[:5, :5])
    
    # Esempio: trova il percorso più breve tra due fermate
    if len(analyzer.graph.nodes()) >= 2:
        nodes = list(analyzer.graph.nodes())
        source, target = nodes[0], nodes[10] if len(nodes) > 10 else nodes[-1]
        path, length = analyzer.find_shortest_path(source, target)
        print(f"\nPercorso più breve da {source} a {target}:")
        print(f"  Distanza: {length:.2f}")
        print(f"  Fermate intermedie: {len(path)}")
    
    # Esporta la matrice
    analyzer.export_distance_matrix("matrice_distanze.csv")
    
    # Visualizza la rete
    fig = analyzer.visualize_network()
    plt.savefig("rete_trasporto.png", dpi=300, bbox_inches='tight')
    print("\n✓ Grafico salvato in: rete_trasporto.png")
    
    plt.show()
