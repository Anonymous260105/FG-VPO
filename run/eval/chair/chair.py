#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copied from: https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py

Modified by: Maxlinn + you

Changes:
1) Python3 compatible; supports json/jsonl input
2) Built-in synonyms.txt, the script can run independently
3) Removed BLEU/CIDEr/ROUGE
4) Added Recall (original implementation: average coverage per image)
5) Added caching (pickle) to speed up repeated evaluation
6) Added Precision / Recall_set / F1 (based on global TP/FP/FN set-level metrics)
7) Built-in fallback for pattern.singularize, no need to install pattern
"""

import os
import sys
import nltk
import json

# --- singularize: prefer using pattern.en; if not, use the local fallback ---
try:
    from pattern.en import singularize  # If your environment has pattern, this line will work
except Exception:
    import re
    def singularize(w: str) -> str:
        w = w.lower()
        # Order is important (first ies -> y)
        if re.search(r'[^aeiou]ies$', w):
            return re.sub(r'ies$', 'y', w)
        if re.search(r'(ses|xes|zes|ches|shes)$', w):
            return re.sub(r'es$', '', w)
        if w.endswith('s') and not w.endswith('ss'):
            return w[:-1]
        return w
# ------------------------------------------------------------

import argparse
import tqdm
import pickle
from collections import defaultdict


# copied from: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
synonyms_txt = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake, cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''

def combine_coco_captions(annotation_path):
    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'val')))
    train_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'train')))
    all_caps = {'info': train_caps['info'],
                'licenses': train_caps['licenses'],
                'images': val_caps['images'] + train_caps['images'],
                'annotations': val_caps['annotations'] + train_caps['annotations']}
    return all_caps 

def combine_coco_instances(annotation_path):
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'val')))
    train_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'train')))
    all_instances = {'info': train_instances['info'],
                     'licenses': train_instances['licenses'],
                     'type': train_instances['licenses'],
                     'categories': train_instances['categories'],
                     'images': train_instances['images'] + val_instances['images'],
                     'annotations': val_instances['annotations'] + train_instances['annotations']}
    return all_instances 


class CHAIR(object):
    def __init__(self, coco_path):
        self.imid_to_objects = defaultdict(list)  # later become a dict of sets
        self.coco_path = coco_path

        # read in synonyms
        synonyms = synonyms_txt.splitlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = []  # mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # hard-coded rules
        coco_double_words = [
            'motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal',
            'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball',
            'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone',
            'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer',
            'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track'
        ]
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        vehicle_words = ['jet', 'train']

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict[f'baby {animal_word}'] = animal_word
            self.double_word_dict[f'adult {animal_word}'] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict[f'passenger {vehicle_word}'] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

        self.get_annotations()

    def _load_generated_captions_into_evaluator(self, cap_file, image_id_key, caption_key):
        """Read predictions (json/jsonl) into memory."""
        self.caps, self.eval_imids = load_generated_captions(cap_file, image_id_key, caption_key)
        assert len(self.caps) == len(self.eval_imids)

    def caption_to_words(self, caption):
        """
        Input: caption
        Output: (all MSCOCO words in caption, their node words, indices, raw double_words list)
        """
        # tokenize & singularize
        words = nltk.word_tokenize(caption.lower())
        words = [singularize(w) for w in words]

        # replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = ' '.join(words[i:i+2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # "toilet seat" special case
        if ('toilet' in words) and ('seat' in words):
            words = [word for word in words if word != 'seat']

        # keep only words in mscoco_objects
        idxs = [idxs[idx] for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]

        node_words = [self.inverse_synonym_dict[word] for word in words]
        return words, node_words, idxs, double_words

    def get_annotations_from_segments(self):
        """Add objects from MSCOCO instance segmentation."""
        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments['annotations']

        id_to_name = {cat['id']: cat['name'] for cat in coco_segments['categories']}

        for i, annotation in enumerate(segment_annotations):
            sys.stdout.write(f"\rGetting annotations for {i}/{len(segment_annotations)} segmentation masks")
            imid = annotation['image_id']
            node_word = self.inverse_synonym_dict[id_to_name[annotation['category_id']]]
            self.imid_to_objects[imid].append(node_word)
        print("\n")

    def get_annotations_from_captions(self):
        """Add objects from MSCOCO ground-truth captions."""
        coco_caps = combine_coco_captions(self.coco_path)
        caption_annotations = coco_caps['annotations']

        for i, annotation in enumerate(caption_annotations):
            sys.stdout.write(f'\rGetting annotations for {i}/{len(coco_caps["annotations"])} ground truth captions')
            imid = annotation['image_id']
            _, node_words, _, _ = self.caption_to_words(annotation['caption'])
            self.imid_to_objects[imid].extend(node_words)
        print("\n")

    def get_annotations(self):
        """Build GT objects from both segmentation & captions."""
        self.get_annotations_from_segments()
        self.get_annotations_from_captions() # Value is too small
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def compute_chair(self, cap_file, image_id_key, caption_key):
        """
        Compute CHAIRs/CHAIRi + Recall; also adds Precision/Recall_set/F1.
        """
        self._load_generated_captions_into_evaluator(cap_file, image_id_key, caption_key)

        imid_to_objects = self.imid_to_objects
        caps = self.caps
        eval_imids = self.eval_imids

        num_caps = 0.0
        num_hallucinated_caps = 0.0
        hallucinated_word_count = 0.0
        coco_word_count = 0.0

        # add: for original Recall (average coverage per image)
        num_recall_gt_objects = 0.0
        num_gt_objects = 0.0

        # add: for Precision / F1 (set-level, micro average)
        TP = 0.0
        FP = 0.0
        FN = 0.0

        output = {'sentences': []}

        for i in tqdm.trange(len(caps)):
            cap: str = caps[i]
            imid: int = eval_imids[i]

            words, node_words, idxs, raw_words = self.caption_to_words(cap)
            gt_objects = imid_to_objects[imid]  # set

            cap_dict = {
                'image_id': imid,
                'caption': cap,
                'mscoco_hallucinated_words': [],
                'mscoco_gt_words': list(gt_objects),
                'mscoco_generated_words': list(node_words),
                'hallucination_idxs': [],
                'words': raw_words
            }
            cap_dict['metrics'] = {'CHAIRs': 0, 'CHAIRi': 0.0, 'Recall': 0.0}

            # count hallucinated words / collect recall set
            coco_word_count += len(node_words)
            hallucinated = False
            recall_gt_objects = set()

            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1
                    cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append(idx)
                    hallucinated = True
                else:
                    recall_gt_objects.add(node_word)

            # sentence-level CHAIRs
            num_caps += 1
            if hallucinated:
                num_hallucinated_caps += 1

            # original Recall accumulation
            num_gt_objects += len(gt_objects)
            num_recall_gt_objects += len(recall_gt_objects)

            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.0
            cap_dict['metrics']['Recall'] = 0.0

            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words']) / float(len(words))
            if len(gt_objects) > 0:
                cap_dict['metrics']['Recall'] = len(recall_gt_objects) / len(gt_objects)

            output['sentences'].append(cap_dict)

            # add: global TP/FP/FN for Precision/F1
            pred_set = set(node_words)   # deduplicated generated object set
            gt_set = set(gt_objects)
            TP += len(pred_set & gt_set)
            FP += len(pred_set - gt_set)
            FN += len(gt_set - pred_set)

        chair_s = (num_hallucinated_caps / num_caps) if num_caps > 0 else 0.0
        chair_i = (hallucinated_word_count / coco_word_count) if coco_word_count > 0 else 0.0
        recall = (num_recall_gt_objects / num_gt_objects) if num_gt_objects > 0 else 0.0

        # add: set-level Precision / Recall_set / F1 (micro average)
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec2 = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * prec * rec2) / (prec + rec2) if (prec + rec2) > 0 else 0.0

        output['overall_metrics'] = {
            'CHAIRs': chair_s,
            'CHAIRi': chair_i,
            'Recall': recall,        # original implementation: average coverage per image
            'Precision': prec,       # added: set-level Precision
            'Recall_set': rec2,      # added: set-level Recall
            'F1': f1                 # added: F1
        }
        return output


def load_generated_captions(cap_file, image_id_key: str, caption_key: str):
    """Load predictions list[dict] from json or jsonl."""
    ext = os.path.splitext(cap_file)[-1].lower()
    if ext == '.json':
        caps = json.load(open(cap_file, 'r', encoding='utf-8'))
    elif ext == '.jsonl':
        caps = [json.loads(s) for s in open(cap_file, 'r', encoding='utf-8')]
    else:
        raise ValueError(f'Unsupported extension {ext} for cap_file: {cap_file}')

    imids = [obj[image_id_key] for obj in caps]
    caps = [obj[caption_key] for obj in caps]
    return caps, imids


def save_hallucinated_words(cap_file, cap_dict):
    with open(cap_file, 'w', encoding='utf-8') as f:
        json.dump(cap_dict, f, indent=2, ensure_ascii=False)


def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    for k, v in sentence_metrics.items():
        k_str = str(k).ljust(12)
        v_str = f'{v * 100:.01f}'
        print(k_str, v_str, sep=': ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cap_file", type=str, default='',
                        help="json/jsonl with [{'image_id':..., 'caption':...}, ...]")
    parser.add_argument("--image_id_key", type=str, default="image_id",
                        help="key holding image id in each record")
    parser.add_argument("--caption_key", type=str, default="caption",
                        help="key holding caption in each record")

    parser.add_argument("--cache", type=str, default="chair.pkl",
                        help="pickle cache for evaluator (speeds up next runs)")
    parser.add_argument("--coco_path", type=str, default='coco_annotations',
                        help="folder that contains *train/val 2014 annotations (first run only)")

    parser.add_argument("--save_path", type=str, default="",
                        help="save detailed results json (optional)")

    args = parser.parse_args()

    if args.cache and os.path.exists(args.cache):
        evaluator = pickle.load(open(args.cache, 'rb'))
        print(f"loaded evaluator from cache: {args.cache}")
    else:
        print("cache not set or not found, building evaluator from scratch...")
        evaluator = CHAIR(args.coco_path)
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        pickle.dump(evaluator, open(args.cache, 'wb'))
        print(f"cached evaluator to: {args.cache}")

    cap_dict = evaluator.compute_chair(args.cap_file, args.image_id_key, args.caption_key)
    print_metrics(cap_dict)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        save_hallucinated_words(args.save_path, cap_dict)
