import React from 'react';
import {render} from 'react-dom';
import {Helmet} from "react-helmet";

import Header from "../components/header.jsx";
import Overview from "../components/overview.jsx";
import Video from "../components/video.jsx";
import Method from "../components/method.jsx";
import Result from "../components/results.jsx";
import Contact from "../components/contact.jsx";
import Footer from "../components/footer.jsx";
import Citation from "../components/citation.jsx";
import ForkMeOnGitHub from "fork-me-on-github";

import data from "../data/template.yaml";

class Template extends React.Component {
  render() {
    return (
      <div>
        <Helmet
          title={data.title}
          meta={[
            { name: 'viewport', content: 'width=device-width,initial-scale=1'},
            { property: "og:site_name", content: data.organization },
            { property: "og:type", content: "article" },
            { property: "og:title", content: data.title },
            { property: "og:description", content: data.description},
            { property: 'og:image', content: data.image}, // FIXME
            { property: "og:image:width", content: "912"},
            { property: "og:image:height", content: "618"},
            { property: 'og:url', content: data.url},
            { name: 'twitter:card', content: "summary_large_image"},
            { name: 'twitter:title', content: data.title},
            { name: 'twitter:image', content: data.image},
            { name: 'twitter:description', content: data.description},
            { name: 'twitter:url', content: data.url},
            { name: 'twitter:site', content: data.twitter},
          ]}
        />
        <div data-uk-sticky className="uk-visible@s">
          <ForkMeOnGitHub
            repo={data.repo}
            colorBackground="#999"
            colorOctocat="white"
          />
        </div>
        <Header title={data.title}
                conference={data.conference}
                authors={data.authors}
                affiliations={data.affiliations}
                meta={data.meta}
                code={data.repo}
                video={data.video}
                paper={data.paper}
                blog={data.blog}  />
        <div className="uk-container uk-container-small">
          <Overview overview={data.overview} />
          <Video video={data.video} />
          <Method method={data.method} />
          <Result results={data.results} />
          <Contact authors={data.authors} contact_ids={data.contact_ids}/>
          <Citation bibtex={data.bibtex} />
        </div>
        <Footer />
      </div>
    )
  }
}

render(<Template/>, document.getElementById('root'))
