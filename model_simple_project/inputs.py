# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import io

from google.cloud import storage
from zipfile import ZipFile


def load_data_from_gs(args):
    bucket_name = args.root_dir
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob("train.zip")
    obj_bytes = blob.download_as_string()

    archive = io.BytesIO()
    archive.write(obj_bytes)

    with ZipFile(archive, "w") as zip_archive:
        zip_archive.extractall(".")


def save_model(args):
    """Saves the model to Google Cloud Storage

    Args:
      args: contains name for saved model.
    """
    scheme = "gs://"
    bucket_name = args.job_dir[len(scheme) :].split("/")[0]

    prefix = "{}{}/".format(scheme, bucket_name)
    bucket_path = args.job_dir[len(prefix) :].rstrip("/")

    datetime_ = datetime.datetime.now().strftime("model_%Y%m%d_%H%M%S")

    if bucket_path:
        model_path = "{}/{}/{}".format(bucket_path, datetime_, args.model_name)
    else:
        model_path = "{}/{}".format(datetime_, args.model_name)

    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(args.model_name)
