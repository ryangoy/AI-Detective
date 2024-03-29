import React from "react"

export default class Spinner extends React.Component {

    render() {
        if(this.props.stage === 'complete' || this.props.stage === 'finished') {
            return <img className="row" src={process.env.PUBLIC_URL + "spinner.gif"}  alt="Spinner gif"/>
        } else {
            return (<div>
                      <img className="row" src={process.env.PUBLIC_URL + "spinner.gif"}  alt="Spinner gif"/>
                      <p>Running <b>{this.props.stage}</b> </p>
                    </div>
                    )
        }
    }
}