import React, { Component } from "react";
import axios from 'axios';

import { Progress } from 'reactstrap';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css'
import Pipeline from "./components/Pipeline"

import openSocket from 'socket.io-client'

const socket = openSocket('http://0.0.0.0:8000');

class App extends Component {

  constructor(props) {
    super(props);
    this.state = {
      selectedFile: null,
      fileField: 'Upload File Here',
      loaded:0,
      percent:-1,
      show_file_upload_progress: false,
      show_spinner: false,
      show_result: false,
      stage: 'none',
      completed_stages: []
    }
    this.subscribeToPipelineUpdates();
  }


  subscribeToPipelineUpdates = () => {
    socket.on('stage', (s) => {
        

        if (this.state.stage !== 'none') {
            this.setState({completed_stages:this.state.completed_stages.concat([this.state.stage])})
        } else {
            this.setState({show_spinner:true})
        }
        this.setState({stage:s})
        if (s === 'completed') {
            this.setState({show_result:true})
        }

    });
  } 

  // when a file is selected
  onChangeHandler = (event) =>{
    if(this.checkFileType(event)){ 
      this.setState({
        selectedFile: event.target.files[0],
        fileField: event.target.files[0].name
      })
    }
  }

  // when we press the upload button
  onClickHandler = () => {
    if (this.state.selectedFile == null) {
      console.log('File is null')
    }
    else {
      this.setState({show_file_upload_progress: true,
                     show_spinner: true,
                     stage: 'none',
                     completed_stages: []})
      const data = new FormData() 
      data.append('file', this.state.selectedFile)

      axios.post("http://0.0.0.0:8000/dev/face_percent", data, {
        
        onUploadProgress: ProgressEvent => {
          this.setState({
               loaded: (ProgressEvent.loaded / ProgressEvent.total*100),
          })
        }

      }).then(res => {
        const percent = res.data['percent']*100;
        return percent
      }).then(percent => {
        this.setState({percent, show_spinner:false});
      })

    }
  }

  // ensure it is an image
  checkFileType= (event) =>{

    let files = event.target.files[0]
    let err = ''
    const types = ['video/mp4']

    if (types.every(type => files.type !== type)) {
      err = files.type+' is not a supported format\n';
    }

    if (err !== '') {
        event.target.value = null
        toast(err)
        return false; 
    }
    return true;

  }


  render() { return (
      <div>
        <form action="/upload" method="POST" encType="multipart/form-data">
          <div className="input-group col-md-5 center-div">
            <div className="custom-file">
              <input type="file" className="custom-file-input" id="inputGroupFile02" onChange={this.onChangeHandler}
                aria-describedby="inputGroupFileAddon01"/>
              <label className="custom-file-label">{this.state.fileField}</label>
            </div>
            <div className="input-group-append">
              <button type="button" className="input-group-append btn btn-primary btn-block" onClick={this.onClickHandler}>Upload</button>
            </div>
          </div>
        </form>
        <div className="form-group">
         <ToastContainer />
        </div>

        { this.state.show_file_upload_progress ?
          <div className="form-group col-md-5 center-div mt-3">
              <Progress max="100" color="success" value={this.state.loaded} >{Math.round(this.state.loaded,2) }%</Progress>
          </div>
          :
          <div></div>
        }
        <div>
        <Pipeline stage={this.state.stage} 
                  completed_stages={this.state.completed_stages}
                  show_spinner={this.state.show_spinner}
                  show_result={this.state.show_result}
                  percent={this.state.percent}/>
        </div>
          
          

      </div>
    )
  }
}

export default App;
