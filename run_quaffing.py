import argparse
import os
import quafing as q



def parse_cla():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", help="filepath of input data", type=str)
    parser.add_argument("--handle", "-h", help="handle for output",type=str)
    parser.add_argument("--outputdirectory","-o", help="directory for output", type=str )
    args = parser.parse_args()
    return args


def check_io_structure(args):
    if not os.path.isfile(args.file):
        raise(FileExistsError,"No input file found")
    
    if not os.path.isdir(args.outputdirectory):
        print(f'creating directory {args.outputdirectory}')
        os.makedirs(args.outputdirectory)

def preprocess_data(rawdata,rawmetadata):
    prep = q.PreProcessor(rawdata,rawmetadata) #create preprocessor instance with data
    prep.select_columns(cols=['e'],deselect=True) #deselect excluded columns
    prep.set_cont_disc() #set which colums contain continuous or discrete data, resp.
    prep.set_density_method(method='Discrete1D',cols=['o','u','b']) #set discretization method
    prep.split_to_groups(0) #split data into groups
    return prep

def make_pdf_collection(prep):
    mdpdfcol = q.create_mdpdf_collection('factorized',prep._groups,prep._grouplabels,prep._groupcolmetadata,)
    mdpdfcol.calculate_distance_matrix(method='hellinger',pwdist='rms')
    mdpdfcol.calculate_shortest_path_matrix()
    return mdpdfcol

def create_embedding(mdpdfcol):
    embedder = q.get_embedder('mds',mdpdfcol)
    embedding = embedder.embed(dimension=2,return_all=True)



    

    











def main():

    args = parse_cla()
    check_io_strucutre(args)

    file = args.file
    rawmetadata, rawdata = q.load(file)

    preprocess_obj = preprocess_data(rawdata,rawmetadata)

    pdf_collection = make_pdf_collection(preprocess_obj)
    
    embedding = create_embedding(pdf_collection)







if __name__ == "__main__":
    main()