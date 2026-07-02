"""Custom datatype for MLIP model files in atlas."""

import os
import tempfile

import torch
from aiida.orm import Data


class MLIPModelFileData(Data):
    """A new data type that contains a MLIP model."""

    def __init__(self, filepath, architecture, **kwargs):
        """Construct a new instance and set the contents to that of the file.

        :param filepath: an absolute filepath of the file to wrap
        :param architecture: the MLIP architecture (e.g. 'mace')
        """
        super().__init__(**kwargs)

        # Get the filename from the absolute path
        filename = os.path.basename(filepath)

        # Store the file in the repository under the given filename
        self.put_object_from_file(filepath, filename)

        # Store in the attributes what the filename is
        self.base.attributes.set('filename', filename)
        self.base.attributes.set('architecture', architecture)

    @property
    def architecture(self):
        """Return the architecture type stored for this instance."""
        return self.base.attributes.get('architecture')

    def load_model(self):
        """Load the model and return it as a pytorch model."""
        curr_architecture = self.base.attributes.get('architecture')

        filename = self.base.attributes.get('filename')
        model_data_bytes = self.get_object_content(filename, mode='rb')

        if curr_architecture.lower() in ['mace', 'scaleshiftmace', 'mace_ase', '']:
            # Write the bytes to a temporary file and load it back
            with tempfile.NamedTemporaryFile(delete=True, suffix='.model') as temp_file:
                temp_file.write(model_data_bytes)
                temp_file.flush()
                temp_file.seek(0)
                model = torch.load(temp_file)
        else:
            raise NotImplementedError('Only MACE models are allowed for now.')

        return model
