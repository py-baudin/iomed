import re
import pathlib

def load(obj):
    return Labels.load(obj)

def save(filename, labels):
    if not isinstance(labels, Labels):
        raise ValueError(f'Expected Labels object, not {type(labels)}')
    labels.save(filename)


def init(num):
    indices = list(range(num))
    descr = [f"Label {i + 1}" for i in range(num)]
    return Labels(indices, descr)


class Labels:
    """dict-like Label container"""

    RE_LABEL = re.compile(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\.\d]+)\s+(\d)\s+(\d)\s+"([\w\s\-]+)"$')

    def __init__(self, indices, descriptions, colors=None, transparency=None, visibility=None):
        self.indices = list(map(int, indices))
        nlabels = len(self.indices)
        assert len(descriptions) == nlabels
        self.descriptions = list(map(str, descriptions))
        if colors is not None:
            assert len(colors) == nlabels
            self.colors = list(tuple(map(int, color)) for color in colors)
        else:
            self.colors = [tuple(np.random.randint(0, 255, 3)) for _ in range(nlabels)]
        if transparency is not None:
            assert len(transparency) == nlabels
            self.transparency = list(map(float, transparency))
        else:
            self.transparency = [1] * nlabels
        if visibility is not None:
            assert len(visibility) == nlabels
            self.visibility = list(map(int, visibility))
        else:
            self.visibility = [1] * nlabels

    def __repr__(self):
        return f"Labels({len(self)})"

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, item):
        if isinstance(item, int):
            dct = dict(zip(self.indices, self.descriptions))
        elif isinstance(item, str):
            unique = set()
            dct = {d: i for d, i in zip(self.descriptions,self.indices) if not (d in unique or unique.add(d))}
        else:
            raise ValueError(f"Invalid item type: {item}")
        return dct[item]

    def append(self, description, *, color=None, transparency=1, visibility=1):
        if color is None:
            color = np.random.randint(0, 255, 3)
        color = tuple(color)
        self.indices.append(max(self.indices) + 1)
        self.descriptions.append(str(description))
        self.colors.append(color)
        self.transparency.append(transparency)
        self.visibility.append(visibility)

    def remove(self, item, reindex=True):
        if isinstance(item, int):
            index = self.indices.index(item)
        if isinstance(item, str):
            index = self.descriptions.index(item)
        else:
            raise ValueError(f"Invalid item type: {item}")
        indices = [i for i in self.indices if i != index]
        return self.subset(indices, reindex=reindex)

    def subset(self, indices, reindex=True):
        num = len(indices)
        true_indices = {self.indices.index(i) for i in indices}
        return Labels(
            list(range(num)) if reindex else indices,
            [self.descriptions[i] for i in true_indices],
            [self.colors[i] for i in true_indices],
            [self.transparency[i] for i in true_indices],
            [self.visibility[i] for i in true_indices],
        )

    @classmethod
    def load(cls, file):
        indices, descr, colors, transp, visib = [], [], [], [], []
        with open(file, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # parse line
                match = cls.RE_LABEL.match(line)
                if not match:
                    raise ValueError(f"Invalid syntax in file {file}: {line}")
                idx, r, g, b, a, v, m, d = match.groups()
                indices.append(int(idx))
                colors.append((int(r), int(g), int(b)))
                transp.append(float(a))
                visib.append(int(v))
                descr.append(d)
        return Labels(indices, descr, colors, transp, visib)

    def save(self, file):
        with open(file, "w") as fp:
            fp.write(self.HEADER)
            for i in range(len(self)):
                idx = self.indices[i]
                r, g, b = self.colors[i]
                a = self.transparency[i]
                v = self.visibility[i]
                d = self.descriptions[i]
                line = f'{idx:5d} {r:5d} {g:5d} {b:5d} {a:9.2f} {v:2d} {1:2d}    "{d}"\n'
                fp.write(line)

    HEADER = """################################################
# Label Description File
# File format: 
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields:
#    IDX:   Zero-based index 
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    MSH:   Label mesh visibility (0 or 1)
#  LABEL:   Label description 
################################################
"""
