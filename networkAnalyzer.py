import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import folium
from folium import plugins

class TransportNetworkAnalyzer:
    """
    Classe per analizzare reti di trasporto pubblico con calcolo probabilità evasione.
    Supporta sia percorsi punto-punto che ispezioni complete (visit all nodes).
    """
    
    def __init__(self, fermate_file, archi_file, probabilita_file, id_column='CODICE_PAL'):
        """
        Inizializza l'analizzatore caricando i file shapefile e il file di probabilità.
        
        Args:
            fermate_file: percorso al file shapefile delle fermate (punti)
            archi_file: percorso al file shapefile degli archi (polyline)
            probabilita_file: percorso al file Excel contenente le probabilità per gli archi
            id_column: nome della colonna da usare come ID delle fermate
        """
        print("Caricamento dei dati...")
        self.fermate = gpd.read_file(fermate_file)
        self.archi = gpd.read_file(archi_file)
        self.probabilita_df = pd.read_excel(probabilita_file)
        self.graph = None
        self.distance_matrix = None
        self.node_names = None
        self.id_column = id_column
        
        print(f"✓ Caricate {len(self.fermate)} fermate")
        print(f"✓ Caricati {len(self.archi)} archi")
        print(f"✓ Caricata tabella probabilità: {len(self.probabilita_df)} righe")
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
        
        print("\n" + "="*60)
        print("STRUTTURA TABELLA PROBABILITÀ")
        print("="*60)
        print(f"\nColonne disponibili: {list(self.probabilita_df.columns)}")
        print(f"\nPrime righe:")
        print(self.probabilita_df.head())
    
    def build_graph(self, epsilon=1e-10):
        """
        Costruisce un grafo NetworkX dalla rete di trasporto.
        Aggiunge sia 'cost' (per Dijkstra) che 'risk' (per massimizzazione evasori).
        """
        print("\nCostruzione del grafo...")
        self.graph = nx.Graph()

        fermata_id_map = {}

        # Aggiungi le fermate come nodi
        for idx, fermata in self.fermate.iterrows():
            node_id = fermata[self.id_column]
            fermata_id_map[idx] = node_id

            self.graph.add_node(
                node_id,
                pos=(fermata.geometry.x, fermata.geometry.y),
                lat=fermata.geometry.y,
                lon=fermata.geometry.x,
                **fermata.to_dict()
            )

        # Aggiungi gli archi
        edges_added = 0
        edges_without_prob = 0

        for idx, arco in self.archi.iterrows():
            line = arco.geometry

            if isinstance(line, LineString):
                start_point = Point(line.coords[0])
                end_point = Point(line.coords[-1])

                start_distances = self.fermate.geometry.distance(start_point)
                start_idx = start_distances.idxmin()
                start_node = fermata_id_map[start_idx]

                end_distances = self.fermate.geometry.distance(end_point)
                end_idx = end_distances.idxmin()
                end_node = fermata_id_map[end_idx]

                # Recupera probabilità
                arco_id = arco['PIPPO']
                prob_rows = self.probabilita_df[self.probabilita_df['Arco'] == arco_id]

                if not prob_rows.empty:
                    p = prob_rows.iloc[0]['Somma di Rischio']
                else:
                    p = 0.01
                    edges_without_prob += 1

                # Clamp probabilità
                p_clamped = float(np.clip(p, 0.0, 1.0))

                # ✅ STEP 1: Aggiungi RISK come metrica additiva
                # risk = -log(1 - p) è crescente con p e additiva
                risk = float(-np.log(max(1e-12, 1.0 - p_clamped)))

                # Cost = lunghezza (per Dijkstra)
                cost = float(line.length)

                if start_node != end_node:
                    edge_attrs = arco.to_dict()
                    edge_attrs.update({
                        'cost': cost,
                        'probability': p_clamped,
                        'risk': risk,  # ✅ Aggiunto
                        'arco_id': arco_id,
                        'geometry': line
                    })

                    self.graph.add_edge(start_node, end_node, **edge_attrs)
                    edges_added += 1

        print(f"✓ Grafo costruito: {self.graph.number_of_nodes()} nodi, {self.graph.number_of_edges()} archi")
        if edges_without_prob > 0:
            print(f"⚠ {edges_without_prob} archi senza probabilità (usata prob. default 0.01)")

        # Verifica connettività
        if nx.is_connected(self.graph):
            print("✓ Il grafo è connesso")
        else:
            components = list(nx.connected_components(self.graph))
            print(f"⚠ Il grafo NON è connesso: {len(components)} componenti separate")
            print(f"  Componente più grande: {len(max(components, key=len))} nodi")

    def _path_stats(self, H, path):
        """
        ✅ STEP 2: Calcola statistiche lungo un percorso.
        
        Returns:
            tuple: (cost_sum, risk_sum, prod_no, edges_info)
        """
        cost_sum = 0.0
        risk_sum = 0.0
        prod_no = 1.0
        edges_info = []
        
        for i in range(len(path) - 1):
            edge_data = H[path[i]][path[i+1]]
            
            c = edge_data.get('cost', 0)
            r = edge_data.get('risk', 0)
            p = edge_data.get('probability', 0)
            
            cost_sum += c
            risk_sum += r
            prod_no *= (1.0 - p)
            
            edges_info.append({
                'from': path[i],
                'to': path[i+1],
                'probability': p,
                'cost': c,
                'risk': r,
                'geometry': edge_data.get('geometry', None)
            })
        
        return cost_sum, risk_sum, prod_no, edges_info

    def _best_next_node_greedy(self, H, current, unvisited):
        """
        ✅ STEP 3: Trova il prossimo nodo migliore con euristica greedy.
        
        Score = risk_sum / cost_sum (massimizza rischio per unità di costo)
        
        Returns:
            tuple: (best_node, best_path, best_score)
        """
        best_node = None
        best_path = None
        best_score = -float('inf')
        
        for u in unvisited:
            try:
                # Trova shortest path in cost da current a u
                path = nx.shortest_path(H, current, u, weight='cost')
                
                # Calcola statistiche
                cost_sum, risk_sum, _, _ = self._path_stats(H, path)
                
                # Score = risk / cost (vogliamo massimizzare rischio minimizzando costo)
                score = risk_sum / (cost_sum + 1e-9)
                
                if score > best_score:
                    best_score = score
                    best_node = u
                    best_path = path
                    
            except nx.NetworkXNoPath:
                continue
        
        return best_node, best_path, best_score

    def find_inspection_route(self, source, n_nodes=150, criterion='centrality'):
        """
        ✅ STEP 4-5: Trova una route che visita tutti i nodi del sottografo.
        
        Usa euristica greedy: ad ogni step sceglie il nodo non visitato
        che massimizza score = risk/cost.
        
        Args:
            source: nodo di partenza
            n_nodes: numero max di nodi nel sottografo
            criterion: criterio per selezionare il sottografo
            
        Returns:
            dict: risultati con route completa, costo, probabilità, ecc.
        """
        if self.graph is None:
            raise ValueError("Prima devi costruire il grafo con build_graph()")

        print("\n" + "="*60)
        print("CALCOLO INSPECTION ROUTE (VISIT ALL NODES)")
        print("="*60)

        # Trova un target lontano da source per costruire sottografo
        print(f"\nCerca nodo lontano da {source}...")
        distances = nx.single_source_dijkstra_path_length(self.graph, source, weight='cost')
        target = max(distances.items(), key=lambda x: x[1])[0]
        print(f"✓ Target selezionato: {target} (distanza: {distances[target]:.2f})")

        # Crea sottografo connesso
        H = self.select_subgraph_connected(source, target, n_nodes, criterion)

        if H is None:
            return None

        print(f"\n{'='*60}")
        print("ALGORITMO GREEDY: VISIT ALL NODES")
        print(f"{'='*60}")

        # Inizializza
        current = source
        visited = {source}

        # NOTA IMPORTANTE (inspection route)
        # La route che costruiamo è un *cammino reale* nel grafo.
        # Quando ci spostiamo da `current` al prossimo nodo scelto, percorriamo
        # un shortest-path (in costo) che può attraversare nodi già visitati.
        # Per questo:
        #   - `route_nodes` può contenere ripetizioni (numero di "passi"),
        #   - `visited` tiene i nodi unici visitati,
        #   - `route_edges` contiene TUTTI gli archi effettivamente attraversati
        #     (anche ripetuti) e deve essere coerente con `route_nodes`.
        route_nodes = [source]
        route_edges = []
        total_cost = 0.0
        total_risk = 0.0
        prod_no = 1.0
        
        all_nodes = set(H.nodes())
        iteration = 0
        
        print(f"Nodi da visitare: {len(all_nodes)}")
        print(f"Inizio da: {source}\n")

        # Loop greedy
        while visited != all_nodes:
            iteration += 1
            unvisited = all_nodes - visited
            
            if iteration % 10 == 0:
                print(f"Iterazione {iteration}: visitati {len(visited)}/{len(all_nodes)} nodi")
            
            # Trova prossimo nodo migliore
            best_node, best_path, best_score = self._best_next_node_greedy(H, current, unvisited)
            
            if best_node is None:
                print(f"⚠ Nessun nodo raggiungibile da {current}")
                break
            
            # Calcola statistiche del path
            path_cost, path_risk, path_prod, path_edges = self._path_stats(H, best_path)
            
            # Aggiorna route (mantieni il cammino completo per coerenza con gli archi)
            route_nodes.extend(best_path[1:])
            visited.update(best_path)

            # Aggiorna totali
            total_cost += path_cost
            total_risk += path_risk
            prod_no *= path_prod
            route_edges.extend(path_edges)

            # Check coerenza (dopo l'update degli archi)
            if len(route_edges) != len(route_nodes) - 1:
                print(
                    "⚠ WARNING mismatch nodes/edges: route_nodes=",
                    len(route_nodes),
                    "route_edges=",
                    len(route_edges),
                )
            
            current = best_node

        print(f"\n✓ Route completata dopo {iteration} iterazioni")
        print(f"✓ Nodi visitati: {len(visited)}/{len(all_nodes)}")

        # Calcola probabilità finale
        prob_almeno_un_evasore = 1.0 - prod_no

        # ✅ STEP 6: Formato output compatibile con visualizzazione esistente
        # Statistiche "uniche" sul sottografo e sulla route
        unique_stops = len(set(route_nodes))
        # archi unici (grafo non orientato -> normalizza (a,b) = (b,a))
        unique_edge_keys = {
            tuple(sorted((e['from'], e['to'])))
            for e in route_edges
        }
        unique_edges = len(unique_edge_keys)

        results = {
            'path': route_nodes,
            'total_cost': total_cost,
            'total_risk': total_risk,
            'prob_almeno_un_evasore': prob_almeno_un_evasore,
            # ATTENZIONE: questi sono conteggi "di percorrenza" (possono includere ripetizioni)
            'n_stops': len(route_nodes),
            'n_edges': len(route_edges),
            # Conteggi "unici" (più vicini all'intuizione: quante fermate diverse ho visitato)
            'unique_stops': unique_stops,
            'unique_edges': unique_edges,
            'subgraph_n_nodes': H.number_of_nodes(),
            'subgraph_n_edges': H.number_of_edges(),
            'edges': route_edges,
            'subgraph': H,
            'source': source,
            'target': route_nodes[-1],  # ultimo nodo visitato
            'mode': 'inspection'
        }

        self._print_results(results, source, route_nodes[-1])

        return results

    def select_subgraph_connected(self, source, target, n_nodes=100, criterion='centrality'):
        """
        Seleziona un sottografo CONNESSO che contenga source e target.
        """
        if self.graph is None:
            raise ValueError("Prima devi costruire il grafo con build_graph()")
        
        if source not in self.graph.nodes():
            raise ValueError(f"Source {source} non esiste nel grafo")
        if target not in self.graph.nodes():
            raise ValueError(f"Target {target} non esiste nel grafo")
        
        if not nx.has_path(self.graph, source, target):
            raise ValueError(f"Source {source} e target {target} non sono connessi nel grafo")
        
        print(f"\nSelezione sottografo connesso con criterio '{criterion}'...")
        
        if criterion == 'distance_based':
            distances_from_source = nx.single_source_dijkstra_path_length(
                self.graph, source, weight='cost'
            )
            distances_from_target = nx.single_source_dijkstra_path_length(
                self.graph, target, weight='cost'
            )
            
            all_nodes = set(self.graph.nodes())
            node_distances = {}
            for node in all_nodes:
                dist_s = distances_from_source.get(node, float('inf'))
                dist_t = distances_from_target.get(node, float('inf'))
                node_distances[node] = min(dist_s, dist_t)
            
            sorted_nodes = sorted(node_distances.items(), key=lambda x: x[1])
            selected_nodes = [node for node, _ in sorted_nodes[:n_nodes]]
            
        elif criterion == 'shortest_path_expansion':
            shortest_path = nx.shortest_path(self.graph, source, target, weight='cost')
            selected_nodes = set(shortest_path)
            
            remaining = n_nodes - len(selected_nodes)
            if remaining > 0:
                remaining_nodes = set(self.graph.nodes()) - selected_nodes
                subgraph_temp = self.graph.subgraph(remaining_nodes)
                
                if len(subgraph_temp.nodes()) > 0:
                    centrality = nx.degree_centrality(subgraph_temp)
                    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                    additional = [node for node, _ in sorted_nodes[:remaining]]
                    selected_nodes.update(additional)
            
            selected_nodes = list(selected_nodes)
            
        else:  # centrality or degree
            selected_nodes = {source, target}
            
            if criterion == 'centrality':
                scores = nx.betweenness_centrality(self.graph, weight='cost')
            else:
                scores = dict(self.graph.degree())
            
            sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for node, _ in sorted_nodes:
                if len(selected_nodes) >= n_nodes:
                    break
                selected_nodes.add(node)
            
            selected_nodes = list(selected_nodes)
        
        H = self.graph.subgraph(selected_nodes).copy()
        
        print(f"✓ Sottografo H creato: {H.number_of_nodes()} nodi, {H.number_of_edges()} archi")
        
        if nx.is_connected(H):
            print("✓ Il sottografo H è connesso")
        else:
            print("⚠ Sottografo non connesso, estraggo componente principale...")
            components = list(nx.connected_components(H))
            
            source_component = None
            for comp in components:
                if source in comp and target in comp:
                    source_component = comp
                    break
            
            if source_component:
                H = H.subgraph(source_component).copy()
                print(f"✓ Componente connessa estratta: {H.number_of_nodes()} nodi, {H.number_of_edges()} archi")
            else:
                print("✗ ERRORE: source e target in componenti diverse!")
                return None
        
        if not nx.has_path(H, source, target):
            print("✗ ERRORE: source e target non connessi nel sottografo finale!")
            return None
        
        print(f"✓ Verificato: source e target sono connessi in H")
        return H
    
    def find_good_source_target(self, min_distance=5):
        """
        Trova una coppia di nodi (source, target) ben connessi e distanti.
        """
        if self.graph is None:
            raise ValueError("Prima devi costruire il grafo con build_graph()")
        
        print("\nRicerca coppia di nodi ottimale...")
        
        degrees = dict(self.graph.degree())
        high_degree_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        candidates = [node for node, _ in high_degree_nodes[:50]]
        
        best_pair = None
        best_distance = 0
        
        for i, source in enumerate(candidates):
            for target in candidates[i+1:]:
                if nx.has_path(self.graph, source, target):
                    try:
                        path = nx.shortest_path(self.graph, source, target)
                        distance = len(path)
                        
                        if distance >= min_distance and distance > best_distance:
                            best_distance = distance
                            best_pair = (source, target)
                    except:
                        continue
        
        if best_pair:
            source, target = best_pair
            print(f"✓ Trovata coppia ottimale:")
            print(f"  Source: {source}")
            print(f"  Target: {target}")
            print(f"  Distanza (hop): {best_distance}")
            return source, target
        else:
            nodes = list(self.graph.nodes())
            return nodes[0], nodes[1]
    
    def find_path_with_evasion_probability(self, source, target, n_nodes=100, criterion='shortest_path_expansion'):
        """
        Trova il percorso con minimo costo (Dijkstra) e calcola P(>=1 evasore).
        Modalità "shortest path" classica.
        """
        if self.graph is None:
            raise ValueError("Prima devi costruire il grafo con build_graph()")

        print("\n" + "="*60)
        print("CALCOLO PERCORSO SHORTEST PATH")
        print("="*60)

        H = self.select_subgraph_connected(source, target, n_nodes, criterion)

        if H is None:
            return None

        print("\nEsecuzione algoritmo Dijkstra...")
        try:
            path = nx.shortest_path(H, source, target, weight='cost')
            total_cost = nx.shortest_path_length(H, source, target, weight='cost')
            print(f"✓ Percorso trovato con costo totale: {total_cost:.4f}")
        except nx.NetworkXNoPath:
            print(f"✗ ERRORE: Nessun percorso trovato")
            return None

        # Calcola statistiche
        _, total_risk, prod_no, path_edges = self._path_stats(H, path)
        prob_almeno_un_evasore = 1.0 - prod_no

        results = {
            'path': path,
            'total_cost': total_cost,
            'total_risk': total_risk,
            'prob_almeno_un_evasore': prob_almeno_un_evasore,
            'n_stops': len(path),
            'n_edges': len(path_edges),
            'edges': path_edges,
            'subgraph': H,
            'source': source,
            'target': target,
            'mode': 'shortest_path'
        }

        self._print_results(results, source, target)

        return results

    
    def _print_results(self, results, source, target):
        """Stampa i risultati in modo formattato."""
        mode = results.get('mode', 'shortest_path')
        
        print("\n" + "="*60)
        print("RISULTATI")
        print("="*60)
        
        if mode == 'inspection':
            print(f"\nInspection Route partendo da {source}:")
        else:
            print(f"\nPercorso da {source} a {target}:")
            
        print(f"  Fermate attraversate (passaggi): {results['n_stops']}")
        print(f"  Archi attraversati (passaggi):   {results['n_edges']}")

        # Se presenti, mostra anche le cardinalità "uniche" (molto utili in inspection)
        if 'unique_stops' in results:
            print(f"  Fermate uniche visitate:        {results['unique_stops']}")
        if 'unique_edges' in results:
            print(f"  Archi unici attraversati:       {results['unique_edges']}")

        if 'subgraph_n_nodes' in results and 'subgraph_n_edges' in results:
            print(f"  Sottografo:                     {results['subgraph_n_nodes']} nodi, {results['subgraph_n_edges']} archi")
        print(f"  Costo totale: {results['total_cost']:.4f}")
        print(f"  Risk totale: {results.get('total_risk', 0):.4f}")
        print(f"\n  P(≥1 evasore sul percorso) = {results['prob_almeno_un_evasore']:.4%}")
        
        print(f"\nPercorso completo (prime 20 fermate):")
        for i, node in enumerate(results['path'][:20]):
            print(f"  {i+1}. {node}")
        if len(results['path']) > 20:
            print(f"  ... (altre {len(results['path']) - 20} fermate)")
        
        print(f"\nDettaglio archi (top 10 per probabilità):")
        sorted_edges = sorted(results['edges'], key=lambda x: x['probability'], reverse=True)
        for i, edge in enumerate(sorted_edges[:10]):
            print(f"  Arco {i+1}: {edge['from']} → {edge['to']}")
            print(f"    Probabilità: {edge['probability']:.4%}")
            print(f"    Risk: {edge.get('risk', 0):.4f}")
            print(f"    Costo: {edge['cost']:.4f}")
    
    def visualize_path_folium(self, results, output_file='mappa_percorso.html'):
        """
        Visualizza il percorso su una mappa interattiva Folium.
        Funziona sia per shortest path che per inspection route.
        """
        if results is None:
            print("Nessun risultato da visualizzare")
            return None
        
        mode = results.get('mode', 'shortest_path')
        print(f"\nCreazione mappa interattiva Folium (modalità: {mode})...")
        
        path_nodes = results['path']
        H = results['subgraph']
        
        lats = [H.nodes[node]['pos'][1] for node in path_nodes]
        lons = [H.nodes[node]['pos'][0] for node in path_nodes]
        
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
        
        path_coords_latlon = []
        for node in path_nodes:
            x, y = H.nodes[node]['pos']
            lon, lat = transformer.transform(x, y)
            path_coords_latlon.append((lat, lon))
        
        center_lat = np.mean([lat for lat, lon in path_coords_latlon])
        center_lon = np.mean([lon for lat, lon in path_coords_latlon])
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # Fermate del sottografo (grigio)
        for node in H.nodes():
            x, y = H.nodes[node]['pos']
            lon, lat = transformer.transform(x, y)
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color='gray',
                fill=True,
                fillColor='gray',
                fillOpacity=0.3,
                popup=f"Fermata: {node}"
            ).add_to(m)
        
        # Archi del percorso
        # NB: in inspection route il cammino può ripassare su nodi già visti,
        # quindi è più robusto disegnare usando direttamente results['edges'].
        for edge_info in results['edges']:
            prob = float(edge_info.get('probability', 0.0))

            if prob < 0.1:
                color = 'green'
            elif prob < 0.2:
                color = 'orange'
            else:
                color = 'red'

            # Se ho una geometria, uso quella; altrimenti collego from->to con una linea.
            geom = edge_info.get('geometry', None)
            locations = None

            if geom is not None:
                # Gestisci LineString / MultiLineString in modo tollerante
                if geom.geom_type == 'LineString':
                    coords_latlon = []
                    for c in geom.coords:
                        x, y = c[0], c[1]  # supporta anche (x,y,z) / (x,y,z,m)
                        lon, lat = transformer.transform(x, y)
                        coords_latlon.append((lat, lon))
                    if len(coords_latlon) >= 2:
                        locations = coords_latlon
                elif geom.geom_type == 'MultiLineString':
                    # prendi la parte più lunga per evitare spezzoni minuscoli
                    parts = list(geom.geoms)
                    if parts:
                        longest = max(parts, key=lambda g: g.length)
                        coords_latlon = []
                        for c in longest.coords:
                            x, y = c[0], c[1]
                            lon, lat = transformer.transform(x, y)
                            coords_latlon.append((lat, lon))
                        if len(coords_latlon) >= 2:
                            locations = coords_latlon

            if locations is None:
                a = edge_info.get('from')
                b = edge_info.get('to')
                if a in H.nodes and b in H.nodes:
                    ax, ay = H.nodes[a]['pos']
                    bx, by = H.nodes[b]['pos']
                    alon, alat = transformer.transform(ax, ay)
                    blon, blat = transformer.transform(bx, by)
                    locations = [(alat, alon), (blat, blon)]
                else:
                    # fallback estremo: ignora l'arco
                    continue

            folium.PolyLine(
                locations=locations,
                color=color,
                weight=5,
                opacity=0.8,
                popup=(
                    f"Da: {edge_info.get('from','?')}<br>"
                    f"A: {edge_info.get('to','?')}<br>"
                    f"Probabilità: {prob:.2%}<br>"
                    f"Risk: {edge_info.get('risk', 0):.4f}<br>"
                    f"Costo: {edge_info.get('cost', 0):.4f}"
                )
            ).add_to(m)

        
        # Nodi del percorso (rosso)
        for i, (lat, lon) in enumerate(path_coords_latlon):
            node = path_nodes[i]
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.8,
                popup=f"Stop {i+1}: {node}"
            ).add_to(m)
        
        # Evidenzia source (verde) e target (blu)
        source_lat, source_lon = path_coords_latlon[0]
        target_lat, target_lon = path_coords_latlon[-1]
        
        folium.Marker(
            location=[source_lat, source_lon],
            popup=f"START: {results['source']}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        folium.Marker(
            location=[target_lat, target_lon],
            popup=f"END: {results['target']}",
            icon=folium.Icon(color='blue', icon='stop')
        ).add_to(m)
        
        # Legenda
        mode_text = "Inspection Route" if mode == 'inspection' else "Percorso Ottimale"

        unique_lines = ""
        if 'unique_stops' in results:
            unique_lines += f"<p><b>Fermate uniche:</b> {results['unique_stops']}</p>"
        if 'unique_edges' in results:
            unique_lines += f"<p><b>Archi unici:</b> {results['unique_edges']}</p>"
        if 'subgraph_n_nodes' in results and 'subgraph_n_edges' in results:
            unique_lines += f"<p><b>Sottografo:</b> {results['subgraph_n_nodes']} nodi, {results['subgraph_n_edges']} archi</p>"

        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 280px; height: 240px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4 style="margin-top:0">{mode_text}</h4>
        <p><b>Probabilità ≥1 evasore:</b> {results['prob_almeno_un_evasore']:.2%}</p>
        <p><b>Costo totale:</b> {results['total_cost']:.2f}</p>
        <p><b>Risk totale:</b> {results.get('total_risk', 0):.2f}</p>
        <p><b>Fermate (passaggi):</b> {results['n_stops']}</p>
        <p><b>Archi (passaggi):</b> {results['n_edges']}</p>
        {unique_lines}
        <hr>
        <p><span style="color:green">●</span> Prob < 10%</p>
        <p><span style="color:orange">●</span> Prob 10-20%</p>
        <p><span style="color:red">●</span> Prob > 20%</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        m.save(output_file)
        print(f"✓ Mappa salvata in: {output_file}")
        
        return m


# ============================================================================
# MAIN CON MODALITÀ SHORTEST PATH E INSPECTION ROUTE
# ============================================================================

if __name__ == "__main__":
    FERMATE_FILE = "Cartografia base/Fermate/00_FERMATE_2015_05_12_point.shp"
    ARCHI_FILE = "Cartografia base/Archi fermate/01_ARCHI_FERMATE_2015_05_12_polyline.shp"
    PROBABILITA_FILE = "Rischio.xlsx"
    
    analyzer = TransportNetworkAnalyzer(FERMATE_FILE, ARCHI_FILE, PROBABILITA_FILE)
    analyzer.build_graph()
    
    # ✅ STEP 8: Menu con scelta modalità
    print("\n" + "="*60)
    print("MODALITÀ DI ANALISI")
    print("="*60)
    print("1. SHORTEST PATH - Percorso minimo tra due punti")
    print("2. INSPECTION ROUTE - Visita tutti i nodi (greedy)")
    
    while True:
        mode_choice = input("\nScegli modalità (1/2): ").strip()
        
        if mode_choice in ['1', '2']:
            break
        else:
            print("⚠ Scelta non valida, riprova")
    
    # ========================================================================
    # SELEZIONE NODI
    # ========================================================================
    
    if mode_choice == '1':
        # SHORTEST PATH: serve source E target
        print("\n" + "="*60)
        print("SHORTEST PATH - Selezione origine e destinazione")
        print("="*60)
        print("1. Selezione AUTOMATICA (consigliata)")
        print("2. Selezione MANUALE (inserisci codici fermata)")
        
        while True:
            scelta = input("\nScegli modalità (1/2): ").strip()
            
            if scelta == "1":
                print("\nRicerca automatica nodi ottimali...")
                min_dist = input("Distanza minima desiderata (default=8, premi INVIO): ").strip()
                min_dist = int(min_dist) if min_dist else 8
                source, target = analyzer.find_good_source_target(min_distance=min_dist)
                break
                
            elif scelta == "2":
                print("\nSelezione manuale dei nodi")
                sample_nodes = list(analyzer.graph.nodes())[:20]
                print(f"\nEsempi di codici fermata:")
                for i, node in enumerate(sample_nodes, 1):
                    print(f"  {i}. {node}")
                
                while True:
                    source = input("\nInserisci codice fermata ORIGINE: ").strip()
                    if source not in analyzer.graph.nodes():
                        print(f"⚠ ERRORE: '{source}' non esiste nel grafo")
                        continue
                    
                    target = input("Inserisci codice fermata DESTINAZIONE: ").strip()
                    if target not in analyzer.graph.nodes():
                        print(f"⚠ ERRORE: '{target}' non esiste nel grafo")
                        continue
                    
                    if not nx.has_path(analyzer.graph, source, target):
                        print(f"⚠ ERRORE: {source} e {target} non sono connessi!")
                        continue
                    
                    print(f"✓ Nodi validi e connessi!")
                    break
                break
            else:
                print("⚠ Scelta non valida, riprova")
    
    else:
        # INSPECTION ROUTE: serve solo source
        print("\n" + "="*60)
        print("INSPECTION ROUTE - Selezione punto di partenza")
        print("="*60)
        print("1. Selezione AUTOMATICA (nodo più centrale)")
        print("2. Selezione MANUALE (inserisci codice fermata)")
        
        while True:
            scelta = input("\nScegli modalità (1/2): ").strip()
            
            if scelta == "1":
                print("\nCalcolo nodo più centrale...")
                centrality = nx.degree_centrality(analyzer.graph)
                source = max(centrality.items(), key=lambda x: x[1])[0]
                print(f"✓ Nodo selezionato: {source}")
                target = None  # Non serve per inspection
                break
                
            elif scelta == "2":
                print("\nSelezione manuale del nodo di partenza")
                sample_nodes = list(analyzer.graph.nodes())[:20]
                print(f"\nEsempi di codici fermata:")
                for i, node in enumerate(sample_nodes, 1):
                    print(f"  {i}. {node}")
                
                while True:
                    source = input("\nInserisci codice fermata ORIGINE: ").strip()
                    if source not in analyzer.graph.nodes():
                        print(f"⚠ ERRORE: '{source}' non esiste nel grafo")
                        continue
                    print(f"✓ Nodo valido!")
                    target = None
                    break
                break
            else:
                print("⚠ Scelta non valida, riprova")
    
    # ========================================================================
    # PARAMETRI AVANZATI
    # ========================================================================
    
    print("\n" + "-"*60)
    print("PARAMETRI AVANZATI (opzionali)")
    print("-"*60)
    
    n_nodes_input = input(f"Numero max nodi sottografo (default=150, INVIO): ").strip()
    n_nodes = int(n_nodes_input) if n_nodes_input else 150
    
    print("\nCriteri di selezione sottografo:")
    print("  1. shortest_path_expansion (consigliato)")
    print("  2. distance_based")
    print("  3. centrality")
    print("  4. degree")
    
    criterion_choice = input("Scegli criterio (1-4, default=1): ").strip()
    criterion_map = {
        '1': 'shortest_path_expansion',
        '2': 'distance_based',
        '3': 'centrality',
        '4': 'degree'
    }
    criterion = criterion_map.get(criterion_choice, 'shortest_path_expansion')
    
    print(f"\n✓ Parametri: n_nodes={n_nodes}, criterion='{criterion}'")
    
    # ========================================================================
    # ESECUZIONE ANALISI
    # ========================================================================
    
    if mode_choice == '1':
        # SHORTEST PATH
        print(f"\n{'='*60}")
        print(f"ANALISI SHORTEST PATH")
        print(f"{'='*60}")
        print(f"Origine:      {source}")
        print(f"Destinazione: {target}")
        
        results = analyzer.find_path_with_evasion_probability(
            source=source,
            target=target,
            n_nodes=n_nodes,
            criterion=criterion
        )
        
        output_file = f'shortest_path_{source}_to_{target}.html'
        
    else:
        # INSPECTION ROUTE
        print(f"\n{'='*60}")
        print(f"ANALISI INSPECTION ROUTE")
        print(f"{'='*60}")
        print(f"Origine: {source}")
        print(f"Modalità: Visita tutti i nodi del sottografo")
        
        results = analyzer.find_inspection_route(
            source=source,
            n_nodes=n_nodes,
            criterion=criterion
        )
        
        output_file = f'inspection_route_{source}.html'
    
    # ========================================================================
    # VISUALIZZAZIONE
    # ========================================================================
    
    if results is not None:
        analyzer.visualize_path_folium(results, output_file=output_file)
        print(f"\n✓ Mappa salvata: '{output_file}'")
        print(f"  Apri il file nel browser per visualizzare la mappa interattiva!")
        
        # Opzione per esportare risultati
        export = input("\nVuoi esportare i risultati in un file di testo? (s/n): ").strip().lower()
        if export == 's':
            if mode_choice == '1':
                results_file = f'risultati_shortest_{source}_to_{target}.txt'
            else:
                results_file = f'risultati_inspection_{source}.txt'
                
            with open(results_file, 'w', encoding='utf-8') as f:
                mode_name = "SHORTEST PATH" if mode_choice == '1' else "INSPECTION ROUTE"
                f.write(f"RISULTATI ANALISI {mode_name}\n")
                f.write("="*60 + "\n\n")
                f.write(f"Origine:      {source}\n")
                if mode_choice == '1':
                    f.write(f"Destinazione: {target}\n")
                f.write(f"\nFermate attraversate: {results['n_stops']}\n")
                f.write(f"Archi attraversati:   {results['n_edges']}\n")
                f.write(f"Costo totale:         {results['total_cost']:.4f}\n")
                f.write(f"Risk totale:          {results.get('total_risk', 0):.4f}\n")
                f.write(f"P(≥1 evasore):        {results['prob_almeno_un_evasore']:.4%}\n\n")
                
                f.write("PERCORSO COMPLETO:\n")
                f.write("-"*60 + "\n")
                for i, node in enumerate(results['path'], 1):
                    f.write(f"{i:3d}. {node}\n")
                
                f.write("\n\nDETTAGLIO ARCHI (ordinati per probabilità):\n")
                f.write("-"*60 + "\n")
                sorted_edges = sorted(results['edges'], key=lambda x: x['probability'], reverse=True)
                for i, edge in enumerate(sorted_edges, 1):
                    f.write(f"\nArco {i}: {edge['from']} → {edge['to']}\n")
                    f.write(f"  Probabilità: {edge['probability']:.4%}\n")
                    f.write(f"  Risk:        {edge.get('risk', 0):.4f}\n")
                    f.write(f"  Costo:       {edge['cost']:.4f}\n")
            
            print(f"✓ Risultati esportati in '{results_file}'")
    
    print("\n" + "="*60)
    print("ANALISI COMPLETATA")
    print("="*60)