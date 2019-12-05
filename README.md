
`docker build .`

`docker run -p 5000:[host port] -v [host data path]:/data [docker image name]`

Tested with command:

`curl -H 'Content-Type:application/json' -d "@[path to mmif json]" -X PUT http://localhost:5000/` 

Where the media location designated in the mmif json is in the volume mounted by the container.
Tested with the following mmif json. 

```
{
  "context": "mmif-prototype-0.0.1.jsonld",
  "metadata": {},
  "media": [
    {
      "id": "0",
      "type": "audio-video",
      "location": "/data/cpb-aacip-398-913n6418.h264.mp4",
      "metadata": {}
    }
  ],
  "contains": {},
  "views": []
}
```
