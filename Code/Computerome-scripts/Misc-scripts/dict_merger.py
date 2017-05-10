import pickle
import itertools

def main():
    full_strict_real = {}
    full_gen_real = {}
    full_be_real = {}
    for num in range(27):
        if len(str(num)) < 2:
            strict_name = 'Results/x0'+str(num)+'_strict_real.pkl'
            gen_name = 'Results/x0'+str(num)+'_gen_real.pkl'
            be_name = 'Results/x0'+str(num)+'_be_real.pkl'
        else:
            strict_name = 'Results/x'+str(num)+'_strict_real.pkl'
            gen_name = 'Results/x'+str(num)+'_gen_real.pkl'
            be_name = 'Results/x'+str(num)+'_be_real.pkl'

        strict_real = pickle.load(open(strict_name, 'rb'))
        gen_real = pickle.load(open(gen_name, 'rb'))
        be_real = pickle.load(open(be_name, 'rb'))

        for strict_key, strict_value in strict_real.items():
            for st_val in strict_value:
                full_strict_real.setdefault(strict_key, []).append(st_val)

        for gen_key, gen_value in gen_real.items():
            for gen_val in gen_value:
                full_gen_real.setdefault(gen_key, []).append(gen_val)

        for be_key, be_value in be_real.items():
            for be_val in be_value:
                full_be_real.setdefault(be_key, []).append(be_val)

    gen_additions = {}
    for key, val in full_strict_real.copy().items():
        if len(val) >= 2:
            val.sort()
            val = list(val for val, _ in itertools.groupby(val))
            if len(val) == 1:
                full_strict_real[key] = val
            else:
                gen_additions[key] = val
                del full_strict_real[key]

    full_gen_real.update(gen_additions)

    pickle.dump(full_strict_real, open('Results/human_mentions_strict_real.pkl', 'wb'))
    pickle.dump(full_gen_real, open('Results/human_mentions_gen_real.pkl', 'wb'))
    pickle.dump(full_be_real, open('Results/human_mentions_be_real.pkl', 'wb'))

if __name__ == '__main__':
    main()
