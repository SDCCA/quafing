from quafing.distance import get_ID_measure

def information_distance(mdpdfs1,mdpdfs2,method=None,pwdist=None,dims=None,kwargs_list=None):

    ID = get_ID_measure(mdpdfs1,mdpdfs2,method=method,pwdist=pwdist,dims=dims)
    infodist = ID.calulate_distance(pwdist=pwdist, kwargs_list=kwargs_list)

    return infodist 