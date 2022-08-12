from quafing.embedding import retrieve_embedder

def get_embedding(method,mdpdf_collection=None, dimension=2,**kwargs)
    embedder = retrieve_embedder(method,mdpdf_collection=mdpdf_collection)
    embeding = embedder.embed(dimension=dimension,return_all=True)
    return embedding

def get_embedder(method,mdpdf_collection=None)
    embedder = retrieve_embedder()
    return embedder

def plot_embedding(embedding,mdpdf_collection,color="distance", plot_title=""):
	    
    color_axis = ["distance","partition","metis"]

    if color not in color_axis:
        raise ValueError(
        	f"color must be one of {[col for col in color_axis]}")

	em = embedding[0]
	auxinfo = embedding[1]
	dimension = auxinfo['dimensions']
	if dimension > 3:
	    print('embedings with dimensions greater than 3 are currently not supported')
	    pass
    else:
        dist_matrix = mdpdf_collection.get_distance_matrix()

        G = nx.from_numpy_matrix(dist_matrix)
        part = community.best_partition(G)
        part2 = metis.part_graph(G)

        if dimension == 2:
            x, y = em[:,0], em[:,1]
            if color == "distance":
                cms = (np.average(em[:,0]), np.average(em[:,1]))
                dx = x - cms[0]
                dy = y - cms[1]
                t = np.sqrt(dx**2 + dy**2)
                t = (t - t.min()) * 100.0 / (t.max() - t.min())
            elif color == "partition":
                t = [part[i] for i in range(len(list(part.keys())))]
            else: # metis
                t = part2[1]

            fig, ax = plt.subplots()
            ax.scatter(x, y, c=t, cmap=cm.jet)
            if show_labels:
                labels = mdpdf_collection._labels
                for i, txt in enumerate(labels):
                    ax.annotate(txt[0], (x[i]+0.05, y[i]))

        elif d == 3:
            if color == "distance":
                cms = (np.average(em[:,0]), np.average(em[:,1]), np.average(em[:,2]))
                dx = em[:,0] - cms[0]
                dy = em[:,1] - cms[1]
                dz = em[:,2] - cms[2]
                t = np.sqrt(dx**2 + dy**2 + dz**2)
                t = (t - t.min()) * 100.0 / (t.max() - t.min())
            elif color == "partition":
                t = [part[i] for i in range(len(list(part.keys())))]
            else: # metis
                t = part2[1]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(em[:,0], em[:,1], em[:,2], marker='o', s=200, c=t, cmap=cm.jet)
            if show_labels:
                labels = mdpdf_collection._labels
                #global lbls
                lbls = []
                for i, txt in enumerate(labels):
                    x2, y2, _ = proj3d.proj_transform(em[i,0], em[i,1],em[i,2], ax.get_proj())
                    lbls += [ax.annotate(txt[0], (x2+0.002, y2))]
        else:
            pass
        dmethod = mdpdf_collection._distance_matrix_type
        emethod = auxinfo["embedding_method"]
        plt.title(plot_title+f" dmethod:{dmethod}  emethod:{emethod}")
        plt.show()
