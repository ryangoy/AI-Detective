import React from "react"

export default class Spinner extends React.Component {

    render() {
        if(this.props.stage === 'complete' || this.props.stage === 'done') {
            return (
                      <img className="row" src={process.env.PUBLIC_URL + "spinner.gif"}  alt="Spinner gif"/>
)
        } else{
            return (<div>
                      <img className="row" src={process.env.PUBLIC_URL + "spinner.gif"}  alt="Spinner gif"/>
                      <p>The AI detective is inspecting your video...</p>
                      </div>
)
        }
    }
}