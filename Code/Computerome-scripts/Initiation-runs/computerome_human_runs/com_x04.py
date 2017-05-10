from parse_and_prepare import ProteinProteinInteractionClassifier as PPIC
import time

def main():
    files = ['Medline',
             'x04',
             'human_entities.tsv',
             '9606.protein.actions.v10.txt']
    clf = PPIC(files, full_sen_set=True)
    clf.prepare()
    st = time.time()
    clf.produce_pairs()
    clf.add_real_ids_and_mask()
    print('sentence_addition = %s seconds' % (time.time() - st))


if __name__ == '__main__':
    main()
