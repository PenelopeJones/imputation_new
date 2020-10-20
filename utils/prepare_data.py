import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import pdb


def smiles_to_fingerprints(smiles_list, extra=False):
    """
    Reads in a .csv file and generates RDKit fingerprints, as well as reading in target ligand coordinates

    :param args: system arguments parsed into main - should contain args.csv, args.tgt

    :return: population, tgt_atoms, tg_species
    """
    print('Generating molecules...')

    rdkit_mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    print('...generating Morgan fingerprints...')

    X = []

    for mol in rdkit_mols:
        try:
            X.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        except:
            continue

    X = np.asarray(X)

    if extra:
        rbs = [Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) for mol in rdkit_mols]
        rbs = np.asarray(rbs).reshape(-1, 1)
        pdb.set_trace()
        X = np.concatenate((X, rbs), axis=1)

    return X


def main(args):
    """
    :return:
    X_train = np.load(args.directory + 'X_train.npy')
    X_test = np.load(args.directory + 'X_test.npy')
    X_train, X_test, _ = transform_data(X_train, X_test)
    pca = PCA(n_components=200)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    np.save(args.directory + 'X_pca_train.npy', X_train)
    np.save(args.directory + 'X_pca_test.npy', X_test)
    """

    extra = True

    if args.dataname == 'Excape':
        """
        
        df = pd.read_csv(args.directory + 'excape/excapeml_compound_info.txt', delimiter="\t")
        df2 = pd.read_csv(args.directory + 'excape/activities.txt', delimiter="\t")
        compounds = df2['compound'].unique()
        smiles = df.loc[df['AMBIT_InChIKey'] == compounds]['AMBIT_SMILES']
        targets = df2['target'].unique()
        n_targets = len(targets)
        n_compounds = len(compounds)

        row_index_dict = dict(zip(compounds, list(range(0, n_compounds))))
        column_index_dict = dict(zip(targets, list(range(0, n_targets))))

        activities = np.full((n_compounds, n_targets), np.nan, dtype=np.double)

        for i in range(df2.shape[0]):
            activities[row_index_dict[df2.at[i, 'compound']], column_index_dict[df2.at[i, 'target']]] = df2.at[i, 'activity']

        pdb.set_trace()

        np.save(args.directory + 'Excape_x1_activities.npy', activities)
        np.save(args.directory + 'Excape_x1_smiles.npy', smiles)
        """
        smiles = np.load(args.directory + 'Excape_x1_smiles.npy', allow_pickle=True)
        y = np.load(args.directory + 'Excape_x1_activities.npy', allow_pickle=True)
        pdb.set_trace()
        X = smiles_to_fingerprints(smiles, extra=False)
        pdb.set_trace()
        idx = np.random.permutation(X.shape[0])
        X_train = X[idx[0:int(0.8*X.shape[0])], :]
        X_valid = X[idx[int(0.8*X.shape[0]):int(0.9*X.shape[0])], :]
        X_test = X[idx[int(0.9*X.shape[0]):], :]
        pdb.set_trace()
        y_train = y[idx[0:int(0.8 * X.shape[0])], :]
        y_valid = y[idx[int(0.8 * X.shape[0]):int(0.9 * X.shape[0])], :]
        y_test = y[idx[int(0.9 * X.shape[0]):], :]

        pdb.set_trace()

        X_train = np.concatenate((X_train, y_train), axis=1)
        X_valid = np.concatenate((X_valid, y_valid), axis=1)
        X_test = np.concatenate((X_test, y_test), axis=1)

        pdb.set_trace()

        np.save(args.directory + 'Excape_x1_train_fingerprints.npy', X_train)
        np.save(args.directory + 'Excape_x1_valid_fingerprints.npy', X_valid)
        np.save(args.directory + 'Excape_x1_test_fingerprints.npy', X_test)

    elif args.dataname == 'Probes':
        df = pd.read_csv(args.directory + 'chemical-probes.csv')
        pdb.set_trace()
        smiles = df['SMILES'].values
        pdb.set_trace()
        X = smiles_to_fingerprints(smiles, extra=False)
        pdb.set_trace()
        np.save(args.directory + 'probes_fingerprints.npy', X)

        pdb.set_trace()


    elif args.dataname == 'Kinase':
        df_train = pd.read_csv(args.directory + 'Kinase_training_w_descriptors.csv')
        df_test = pd.read_csv(args.directory + 'Kinase_test_w_descriptors.csv')

        if extra:
            cols = [2, 4, 12, 13, 14, 15]
            extra_train = np.array(df_train)[:, cols]
            extra_test = np.array(df_test)[:, cols]

        X_train1 = np.array(df_train)[:, -159:]
        X_test1 = np.array(df_test)[:, -159:]

        smiles_list_train = df_train['Structure'].values
        X_train = smiles_to_fingerprints(smiles_list_train, extra)

        smiles_list_test = df_test['Structure'].values
        X_test = smiles_to_fingerprints(smiles_list_test, extra)

        X_train = np.concatenate((X_train, X_train1), axis=1)
        X_test = np.concatenate((X_test, X_test1), axis=1)

        pdb.set_trace()

        np.save(args.directory + 'Kinase_x1_train_fingerprintsplus.npy', X_train)
        np.save(args.directory + 'Kinase_x1_test_fingerprintsplus.npy', X_test)

        pdb.set_trace()

    else:
        df = pd.read_csv(args.directory + 'Adrenergic_dataset.csv')
        idx = np.random.permutation(df.shape[0])
        df_train = df.iloc[idx[0:int(0.8*df.shape[0])]]
        df_test = df.iloc[idx[int(0.8*df.shape[0]):]]

        X_train1 = np.array(df_train)[:, -5:]
        X_test1 = np.array(df_test)[:, -5:]

        if extra:
            cols = [2, 4, 12, 13, 14, 15]
            extra_train = np.array(df_train)[:, cols]
            extra_test = np.array(df_test)[:, cols]

            X_train1 = np.concatenate((extra_train, X_train1), axis=1)
            X_test1 = np.concatenate((extra_test, X_test1), axis=1)

        pdb.set_trace()
        smiles_list_train = df_train['Structure'].values
        X_train = smiles_to_fingerprints(smiles_list_train, extra)

        smiles_list_test = df_test['Structure'].values
        X_test = smiles_to_fingerprints(smiles_list_test, extra)

        X_train = np.concatenate((X_train, X_train1), axis=1)
        X_test = np.concatenate((X_test, X_test1), axis=1)

        pdb.set_trace()

        np.save(args.directory + 'Adrenergic_x1_train_fingerprintsplus.npy', X_train)
        np.save(args.directory + 'Adrenergic_x1_test_fingerprintsplus.npy', X_test)

        pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='../experiments/imputation/data/raw_data/',
                        help='Number of training iterations.')
    parser.add_argument('--dataname', default='Adrenergic',
                        help='Number of training iterations.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test fraction.')

    args = parser.parse_args()

    main(args)
