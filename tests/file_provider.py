from pathlib import Path
from typing import List, NoReturn
from dataclasses import dataclass

import fabric
import shutil
import logging


@dataclass
class FileProvider:
    filenames: List[str]
    destdir: str
    cachedir: str
    hostname: str
    remote_path: str
    password: str
    port: int = 22
    clobber: bool = False
    copy_not_link: bool = False

    def get_files(self) -> NoReturn:
        with self._get_fabric_connection() as conn:
            for filename in self.filenames:
                self._get_file(filename, conn)

    def _get_file(self, filename, conn):
        cached_path = Path(self.cachedir, filename)
        dest_path = Path(self.destdir, filename)
        # TODO: Revisar este if. Al copiar un fichero nuevo en testfiles/postprocess
        #  y querer pasarlo al tempdir, entra en el IF y falla (solo la primera vez)
        #  y en cambio en el else si funciona. A partir de la 2 vez ya funciona
        #  entrando en el IF
        if cached_path.exists() and not self.clobber:
            logging.info(
                f"{cached_path} already exists, linking it to {dest_path}"
            )
            self._link_or_copy(cached_path, dest_path)
        else:
            logging.info(f"Downloading {cached_path} for testing")
            remote_path = Path(self.remote_path, filename)
            logging.debug(f"Downloading {remote_path} to {cached_path}")
            conn.get(remote_path, local=str(cached_path))
            self._link_or_copy(cached_path, dest_path)

    def _link_or_copy(self, cached_path, dest_path):
        if self.copy_not_link:
            shutil.copy2(cached_path, dest_path)
        else:
            self._link_if_different(cached_path, dest_path)

    def  _link_if_different(self, cached_path, dest_path):
        try:
            dest_path.symlink_to(cached_path)
            logging.debug(f"Linked {dest_path} to {cached_path}")
        except FileExistsError:
            pass

    def _get_fabric_connection(self):
        return fabric.Connection(
                self.hostname,
                port=self.port,
                connect_kwargs=dict(password=self.password, allow_agent=False)
        )


def get_remote_file(
        filename: str,
        localdir: str,
        cache_dir: Path = Path("/tmp/aq_biascorrection_test_files"),
        remote_path: str = "testfiles/aq-biascorrection",
        clobber: bool = False,
        copy_not_link: bool = False
) -> str:
    if filename != Path(filename).name:
        # There are subfolders in the file name, we need to take these into
        # account
        actual_filename = Path(filename).name
        extra_dirs = Path(filename).parent
        cache_dir = Path(cache_dir, extra_dirs)
        localdir = Path(localdir, extra_dirs)
        remote_path = Path(remote_path, extra_dirs)
        localdir.mkdir(parents=True, exist_ok=True)
    else:
        actual_filename = filename

    cache_dir.mkdir(parents=True, exist_ok=True)
    ttp = FileProvider(
        filenames=[actual_filename, ],
        destdir=localdir,
        cachedir=str(cache_dir),
        hostname="tester@10.11.12.17",
        remote_path=remote_path,
        port=8222,
        password="8D3zhMi0dr",
        clobber=clobber,
        copy_not_link=copy_not_link
    )
    ttp.get_files()
    return str(Path(localdir, filename))