from parse_and_prepare import ProteinProteinInteractionClassifier as PPIC

def main():
    files = ['Medline',
             'rat_mentions',
             'rat_entities.tsv',
             '10116.protein.actions.v10.txt']
    clf = PPIC(files, full_sen_set=True)
    clf.prepare()
    st = time.time()
    clf.produce_pairs()
    clf.add_real_ids_and_mask()
    print('sentence_addition = %s seconds' % (time.time() - st))


if __name__ == '__main__':
    main()
